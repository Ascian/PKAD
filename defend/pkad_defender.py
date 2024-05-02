from .defender import Defender

import datasets
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from scipy.stats import gaussian_kde

from log_asr_trainer import LogAsrTrainer
from task_pattern import TaskPattern

import time
import logging
from tqdm import tqdm
import os
import shutil
import json

logger = logging.getLogger("root")

class PkadDefender(Defender):
    def __init__(
            self,
            poison_rate=0.15,
            lda_steps_limit=20,
            **kwargs
        ):

        super().__init__(**kwargs)

        self.poison_rate = poison_rate
        self.lda_steps_limit = lda_steps_limit
        
    def defend(self, model, tokenizer, training_args, peft_config, original_datasets, attacker_args, begin_time):
        task_name = attacker_args['data']['task_name']
        poison_name = attacker_args['data']['poison_name']
        
        if os.path.exists(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, attacker_args['model']['model_name_or_path'])):
            shutil.rmtree(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, attacker_args['model']['model_name_or_path']))

        start_train = time.time()
        
        all_dataset = datasets.concatenate_datasets([
                original_datasets['clean_train'],
                original_datasets['clean_validation'],
                original_datasets['poison_train'],
                original_datasets['poison_validation']
                ])
        # data_idxes = np.array(all_dataset['idx'])
        # sentences = np.array(all_dataset['sentence'])
        labels = np.array(all_dataset['label'])
        unique_labels = np.unique(labels)

        poison_mask = np.array([False if i < len(original_datasets['clean_train']) + len(original_datasets['clean_validation']) else True for i in range(len(all_dataset))])
        label_masks = {label: ~poison_mask & (labels == label) for label in unique_labels}
        
        activations = []
        def input_processing(batch):
            inputs = []
            for data in batch['sentence']:
                inputs.append(TaskPattern.get_input(task_name, data))
            return {'input': inputs}
        all_dataset = all_dataset.map(
            input_processing,
            batched=True
            )
        dataloader = DataLoader(
            all_dataset,
            batch_size=training_args.per_device_train_batch_size
            )
        
        activations = [torch.empty(0, model.config.hidden_size).cuda() for _ in range(0,model.config.num_hidden_layers + 1)]
            
        logger.info(f'{time.time() - begin_time} - Compute activations')

        with torch.no_grad():
            for data in tqdm(dataloader, desc="Compute activations", total=len(dataloader)):
                inputs = tokenizer(data['input'], return_tensors="pt", padding='longest').to('cuda')
                outputs = model(**inputs, output_hidden_states=True)

                for hidden_state in range(0, model.config.num_hidden_layers + 1):
                    activations[hidden_state] = torch.cat((activations[hidden_state], outputs.hidden_states[hidden_state][:,-1,:]))
        for hidden_state in range(0, model.config.num_hidden_layers + 1):
            activations[hidden_state] = activations[hidden_state].cpu().detach().numpy()

        biggest_distan_hidden_state = 0
        biggest_distan = -1
        biggest_distan_activation_pca = None
        logger.info(f'{time.time() - begin_time} - Compute mean difference of activations between labels')
        for hidden_state in tqdm(range(1, model.config.num_hidden_layers + 1), desc="Compute mean difference of activations between labels at each hidden state", total=model.config.num_hidden_layers):
            # Activation
            activation_original = activations[hidden_state]

            activation_normalized = StandardScaler().fit_transform(activation_original)

            activation_pca = PCA(n_components=10).fit_transform(activation_normalized)

            activation_tsne = TSNE(n_components=2).fit_transform(activation_pca)

            # Draw the activation with labels
            fig = plt.figure(figsize=(20, 20))
            activation_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            for i in range(len(unique_labels)):
                label = unique_labels[i] 
                mask = label_masks[label]
                label_activation_tsne = activation_tsne[mask]
                plt.scatter(label_activation_tsne[:, 0], label_activation_tsne[:, 1], s=250, facecolors='none', linewidths=3, color=activation_colors[i], label=TaskPattern.get_labels(task_name, label))
            poison_activation_tsne = activation_tsne[poison_mask]
            plt.scatter(poison_activation_tsne[:, 0], poison_activation_tsne[:, 1], s=250, marker='x', linewidth=3, color='b', label='poison')
            ax = plt.gca()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.legend(fontsize=30, loc='upper right')
            os.makedirs(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, attacker_args['model']['model_name_or_path'],  f'hidden_state{hidden_state}'), exist_ok=True)
            plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, attacker_args['model']['model_name_or_path'],  f'hidden_state{hidden_state}', 'labels'))
            plt.close()

            metrics = dict()

            # Euclidean Distance
            euclidean_distan = 0
            label_means = np.array([np.mean(activation_pca[label_masks[label]], axis=0) for label in unique_labels])
            inner_distance = np.array([np.mean(np.linalg.norm(activation_pca[label_masks[label]] - label_means[label], ord=2, axis=1)) for label in unique_labels])
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    outer_distance = np.linalg.norm(label_means[i] - label_means[j], ord=2)
                    euclidean_distan += outer_distance / ((inner_distance[i] + inner_distance[j]) / 2)
            metrics['euclidean_distan'] = euclidean_distan

            if euclidean_distan >= biggest_distan:
                biggest_distan = euclidean_distan
                biggest_distan_hidden_state = hidden_state
                biggest_distan_activation_pca = activation_pca

            metrics_file = os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, attacker_args['model']['model_name_or_path'],  f'hidden_state{hidden_state}', 'metrics.json')
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            with open(metrics_file, 'w') as f:
                metrics = {k: str(v) for k, v in metrics.items()}
                json.dump(metrics, f, indent=4)

            


        logger.info(f'{time.time() - begin_time} - Start filter the samples with wrong label')

        # Mahalanobis Distance
        mahalanobis_distan = np.zeros(0)
        label_means = np.array([np.mean(biggest_distan_activation_pca[label_masks[label]], axis=0) for label in unique_labels])
        label_covariances = np.array([np.cov(biggest_distan_activation_pca[label_masks[label]], rowvar=False) for label in unique_labels])
        for i, activation in enumerate(biggest_distan_activation_pca):
            label_distances = {label: np.sqrt((activation - label_mean).T @ np.linalg.inv(label_covariance) @ (activation - label_mean)) for label, label_mean, label_covariance in zip(unique_labels, label_means, label_covariances)}
            closest_label = min(label_distances, key=label_distances.get)
            if closest_label == labels[i]:
                mahalanobis_distan = np.append(1)
            else:
                mahalanobis_distan = np.append(mahalanobis_distan, label_distances[closest_label] - label_distances[labels[i]])
        
        is_clean = (mahalanobis_distan >= 0)

        negative_values = -mahalanobis_distan[mahalanobis_distan < 0]

        hidden_state = model.config.num_hidden_layers
        poison_tps = []
        filter_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        poison_fps = []
        for filter_rate in filter_rates:
            threshold = np.percentile(negative_values, 100 - filter_rate * 100)
            is_poison = (mahalanobis_distan < 0) & (-mahalanobis_distan >= threshold)
            # Compute the tpr and fpr
            poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()
            poison_tpr = poison_tp / (poison_tp + poison_fn)
            poison_fpr = poison_fp / (poison_fp + poison_tn)
            clean_tn, clean_fp, clean_fn, clean_tp = confusion_matrix(~poison_mask, ~is_poison).ravel()
            clean_tpr = clean_tp / (clean_tp + clean_fn)
            clean_fpr = clean_fp / (clean_fp + clean_tn)
            poison_tps.append(poison_tp)
            poison_fps.append(poison_fp)
            metrics = dict()
            metrics['poison TPR'] = f'{poison_tpr}({poison_tp}/{poison_tp+poison_fn})'
            metrics['poison FPR'] = f'{poison_fpr}({poison_fp}/{poison_fp+poison_tn})'
            metrics['Clean TPR'] = f'{clean_tpr}({clean_tp}/{clean_tp+clean_fn})'
            metrics['Clean FPR'] = f'{clean_fpr}({clean_fp}/{clean_fp+clean_tn})'
            metrics_file = os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, attacker_args['model']['model_name_or_path'],  f'hidden_state{hidden_state}', f'hidden_state{biggest_distan_hidden_state}-mahalanobis-{filter_rate}.json')
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)

        plt.figure(figsize=(10, 10))
        plt.plot(filter_rates, poison_tps, label='poison', color='r')
        plt.plot(filter_rates, poison_fps, label='clean', color='g')
        plt.xlabel('fiter rate')
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, attacker_args['model']['model_name_or_path'],  f'hidden_state{biggest_distan_hidden_state}-mahalanobis-amount'))
        plt.close()
        
        misclassify_rate = np.sum((mahalanobis_distan < 0)) / len(mahalanobis_distan)
        filter_rate = 1 / (1 + ((misclassify_rate ** 2) / self.poison_rate ** 2))
        threshold = np.percentile(negative_values, 100 - filter_rate * 100)
        is_poison = (mahalanobis_distan < 0) & (-mahalanobis_distan >= threshold)

        total_kde = gaussian_kde(-mahalanobis_distan[mahalanobis_distan < 0])
        total_x_range = np.linspace(min(-mahalanobis_distan[mahalanobis_distan < 0]), max(-mahalanobis_distan[mahalanobis_distan < 0]), 500)
        total_kde_values = total_kde(total_x_range)
        poison_kde = gaussian_kde(-mahalanobis_distan[(mahalanobis_distan < 0) & poison_mask])
        poison_kde_values = poison_kde(total_x_range)
        clean_kde = gaussian_kde(-mahalanobis_distan[(mahalanobis_distan < 0) & ~poison_mask])
        clean_kde_values = clean_kde(total_x_range)
        plt.figure(figsize=(10, 10))
        plt.plot(total_x_range, total_kde_values, label='total', color='b')
        plt.plot(total_x_range, poison_kde_values, label='poison', color='r')
        plt.plot(total_x_range, clean_kde_values, label='clean', color='g')
        plt.plot(total_x_range, poison_kde_values - clean_kde_values, label='difference', color='y')
        plt.axvline(x=threshold, color='r', linestyle='--')
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('mahalanobis distance')
        plt.ylabel('density')
        plt.ylim(0)
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, attacker_args['model']['model_name_or_path'],  f'hidden_state{biggest_distan_hidden_state}-mahalanobis-density'))
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.hist(-mahalanobis_distan[mahalanobis_distan < 0], bins=30, alpha=0.75, color='b', label='total')
        plt.hist(-mahalanobis_distan[(mahalanobis_distan < 0) & poison_mask], bins=30, alpha=0.75, color='r', label='poison')
        plt.hist(-mahalanobis_distan[(mahalanobis_distan < 0) & ~poison_mask], bins=30, alpha=0.75, color='g', label='clean')
        plt.axvline(x=threshold, color='r', linestyle='--')
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('mahalanobis distance')
        plt.ylabel('frequency')
        plt.ylim(0)
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, attacker_args['model']['model_name_or_path'], f'hidden_state{biggest_distan_hidden_state}-mahalanobis-frequency'))
        plt.close()

        original_is_poison = is_poison
        original_is_clean = is_clean

        hidden_state = model.config.num_hidden_layers
        # Compute the tpr and fpr
        poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()
        poison_tpr = poison_tp / (poison_tp + poison_fn)
        poison_fpr = poison_fp / (poison_fp + poison_tn)
        clean_tn, clean_fp, clean_fn, clean_tp = confusion_matrix(~poison_mask, ~is_poison).ravel()
        clean_tpr = clean_tp / (clean_tp + clean_fn)
        clean_fpr = clean_fp / (clean_fp + clean_tn)
        metrics = dict()
        metrics['poison TPR'] = f'{poison_tpr}({poison_tp}/{poison_tp+poison_fn})'
        metrics['poison FPR'] = f'{poison_fpr}({poison_fp}/{poison_fp+poison_tn})'
        metrics['Clean TPR'] = f'{clean_tpr}({clean_tp}/{clean_tp+clean_fn})'
        metrics['Clean FPR'] = f'{clean_fpr}({clean_fp}/{clean_fp+clean_tn})'
        metrics_file = os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, attacker_args['model']['model_name_or_path'],  f'hidden_state{hidden_state}', 'detect', f'hidden_state{biggest_distan_hidden_state}-mahalanobis-step0.json')
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

        is_poison = original_is_poison.copy()
        is_clean = original_is_clean.copy()
        # Activation
        activation = activations[hidden_state]

        last_is_poison = is_poison
        lda_step = 1
        while lda_step <= self.lda_steps_limit:
            poison_activation = activation[is_poison]
            clean_activation = activation[is_clean]

            poison_clean = np.vstack([poison_activation, clean_activation])
            poison_clean_label = np.hstack([np.zeros(len(poison_activation)), np.ones(len(clean_activation))])
            activation_lda_origin = LDA().fit(poison_clean, poison_clean_label)

            activation_lda_prediction = activation_lda_origin.predict(activation)
            activation_lda_prediction = activation_lda_prediction == 0
            is_poison = activation_lda_prediction.copy()
            
            # Compute the tpr and fpr
            poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()
            poison_tpr = poison_tp / (poison_tp + poison_fn)
            poison_fpr = poison_fp / (poison_fp + poison_tn)
            clean_tn, clean_fp, clean_fn, clean_tp = confusion_matrix(~poison_mask, ~is_poison).ravel()
            clean_tpr = clean_tp / (clean_tp + clean_fn)
            clean_fpr = clean_fp / (clean_fp + clean_tn)
            metrics = dict()
            metrics['poison TPR'] = f'{poison_tpr}({poison_tp}/{poison_tp+poison_fn})'
            metrics['poison FPR'] = f'{poison_fpr}({poison_fp}/{poison_fp+poison_tn})'
            metrics['Clean TPR'] = f'{clean_tpr}({clean_tp}/{clean_tp+clean_fn})'
            metrics['Clean FPR'] = f'{clean_fpr}({clean_fp}/{clean_fp+clean_tn})'
            metrics_file = os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, attacker_args['model']['model_name_or_path'],  f'hidden_state{hidden_state}', 'detect', f'hidden_state{biggest_distan_hidden_state}-mahalanobis-step{lda_step}.json')
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            if np.all(is_poison == last_is_poison):
                break
            last_is_poison = is_poison.copy()
            
            lda_step += 1

        is_clean = ~is_poison
        while lda_step <= self.lda_steps_limit:
            poison_activation = activation[is_poison]
            clean_activation = activation[is_clean]

            poison_clean = np.vstack([poison_activation, clean_activation])
            poison_clean_label = np.hstack([np.zeros(len(poison_activation)), np.ones(len(clean_activation))])
            activation_lda_origin = LDA().fit(poison_clean, poison_clean_label)

            activation_lda_prediction = activation_lda_origin.predict(activation)
            activation_lda_prediction = activation_lda_prediction == 0
            is_poison = activation_lda_prediction.copy()
            is_clean = ~is_poison
            
            # Compute the tpr and fpr
            poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()
            poison_tpr = poison_tp / (poison_tp + poison_fn)
            poison_fpr = poison_fp / (poison_fp + poison_tn)
            clean_tn, clean_fp, clean_fn, clean_tp = confusion_matrix(~poison_mask, ~is_poison).ravel()
            clean_tpr = clean_tp / (clean_tp + clean_fn)
            clean_fpr = clean_fp / (clean_fp + clean_tn)
            metrics = dict()
            metrics['poison TPR'] = f'{poison_tpr}({poison_tp}/{poison_tp+poison_fn})'
            metrics['poison FPR'] = f'{poison_fpr}({poison_fp}/{poison_fp+poison_tn})'
            metrics['Clean TPR'] = f'{clean_tpr}({clean_tp}/{clean_tp+clean_fn})'
            metrics['Clean FPR'] = f'{clean_fpr}({clean_fp}/{clean_fp+clean_tn})'
            metrics_file = os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, attacker_args['model']['model_name_or_path'],  f'hidden_state{hidden_state}', 'detect', f'hidden_state{biggest_distan_hidden_state}-mahalanobis-step{lda_step}.json')
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            if np.all(is_poison == last_is_poison):
                break
            last_is_poison = is_poison.copy()
            
            lda_step += 1



        #     # activation_tsne = TSNE(n_components=2).fit_transform(biggest_diff_activation_pca)
        #     # fig = plt.figure(figsize=(20, 20))
        #     # poison_activation_tsne = activation_tsne[is_poison]
        #     # true_poison_activation_tsne = activation_tsne[poison_mask]
        #     # clean_activation_tsne = activation_tsne[is_clean]
        #     # plt.scatter(clean_activation_tsne[:, 0], clean_activation_tsne[:, 1], s=250, facecolors='none', linewidths=3, color='g', label='Clean')
        #     # plt.scatter(poison_activation_tsne[:, 0], poison_activation_tsne[:, 1], s=250, facecolors='none', linewidth=3, color='r', label='poison')
        #     # plt.scatter(true_poison_activation_tsne[:, 0], true_poison_activation_tsne[:, 1], s=250, marker='x', linewidth=3, color='b', label='True poison')
        #     # plt.legend()
        #     # plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, attacker_args['model']['model_name_or_path'],  f'hidden_state{biggest_diff_hidden_state}', f'{clean_rate}-{poison_rate}-{reclean_rate}-poison'))
        #     # plt.close()
        
        # logger.info(f'{time.time() - begin_time} - Finish filter the samples with wrong label')
        
        # activation = activations[model.config.num_hidden_layers] # Use the last hidden state
        # logger.info(f'{time.time() - begin_time} - Start detect poison')
        # original_is_poison = is_poison
        # original_is_clean = is_clean
        # is_poison = original_is_poison.copy()
        # is_clean = original_is_clean.copy()
        # last_is_poison = is_poison
        # lda_step = 1
        # while lda_step <= self.lda_steps_limit:
        #     poison_activation = activation[is_poison]
        #     clean_activation = activation[is_clean]

        #     poison_clean = np.vstack([poison_activation, clean_activation])
        #     poison_clean_label = np.hstack([np.zeros(len(poison_activation)), np.ones(len(clean_activation))])
        #     activation_lda_origin = LDA().fit(poison_clean, poison_clean_label)

        #     activation_lda_prediction = activation_lda_origin.predict(activation)
        #     activation_lda_prediction = activation_lda_prediction == 0
        #     is_poison = activation_lda_prediction.copy()
            
        #     if np.all(is_poison == last_is_poison):
        #         break
        #     last_is_poison = is_poison.copy()
            
        #     lda_step += 1

        # logger.info(f'{time.time() - begin_time} - Finish detect poison')

        # def formatting_func(example):
        #     output_texts = []
        #     for i in range(len(example['sentence'])):
        #         text = TaskPattern.get_input(attacker_args['data']['task_name'], example['sentence'][i], example['label'][i])
        #         output_texts.append(text)
        #     return output_texts
        
        # clean_train_clean_indices = np.where(~is_poison[0:len(original_datasets['clean_train'])])[0]
        # poison_train_clean_indices = np.where(~is_poison[len(original_datasets['clean_train']) + len(original_datasets['clean_validation']):len(original_datasets['clean_train']) + len(original_datasets['clean_validation']) + len(original_datasets['poison_train'])])[0]
        # trainer = LogAsrTrainer(
        #     model=model,
        #     tokenizer=tokenizer,
        #     args=training_args,
        #     train_dataset=datasets.concatenate_datasets([
        #         original_datasets['clean_train'].select(clean_train_clean_indices),
        #         original_datasets['poison_train'].select(poison_train_clean_indices)
        #         ]),
        #     eval_dataset={
        #         'clean': original_datasets['clean_validation'], 
        #         'poison': original_datasets['poison_validation'], 
        #         },
        #     peft_config=peft_config,
        #     formatting_func=formatting_func,
        #     max_seq_length=5000,
        # )

        # logger.info(f'{time.time()-begin_time} - Start training')
        # trainer.train()
        # logger.info(f'{time.time()-begin_time} - Training finished')

        # end_train = time.time()
        
        # start_eval = time.time()
        
        # logger.info(f'{time.time()-begin_time} - Start evaluation')
        
        # acc = Defender.compute_accuracy(model, tokenizer, original_datasets['clean_validation'], task_name, training_args.per_device_eval_batch_size)
        # asr = Defender.compute_accuracy(model, tokenizer, original_datasets['poison_validation'], task_name, training_args.per_device_eval_batch_size)
        
        # logger.info(f'{time.time()-begin_time} - Evaluation finished')

        # end_eval = time.time()

        # # Compute the tpr and fpr
        # poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()
        # poison_tpr = poison_tp / (poison_tp + poison_fn)
        # poison_fpr = poison_fp / (poison_fp + poison_tn)

        return {
            'epoch': training_args.num_train_epochs,
            # 'ASR': asr,
            # 'ACC': acc,
            'TPR': f'{poison_tpr}({poison_tp}/{poison_tp+poison_fn})',
            'FPR': f'{poison_fpr}({poison_fp}/{poison_fp+poison_tn})',
            'lda step': f'{lda_step}',
            # 'train time': end_train - start_train,
            # 'eval time': end_eval - start_eval
        }
