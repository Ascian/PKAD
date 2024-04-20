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
            filter_rate=0.15,
            lda_steps_limit=20,
            **kwargs
        ):

        super().__init__(**kwargs)

        self.filter_rate = filter_rate
        self.lda_steps_limit = lda_steps_limit
        
    def defend(self, model, tokenizer, training_args, peft_config, original_datasets, attacker_args, begin_time):
        task_name = attacker_args['data']['task_name']
        poison_name = attacker_args['data']['poison_name']
        
        if os.path.exists(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name)):
            shutil.rmtree(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name))

        clean_train_index = attacker_args['data']['clean_train_index']
        clean_validation_index = attacker_args['data']['clean_validation_index']
        poison_train_index = attacker_args['data']['poison_train_index']
        poison_validation_index = attacker_args['data']['poison_validation_index']

        all_dataset = datasets.concatenate_datasets([
                original_datasets['clean_train'],
                original_datasets['clean_validation'],
                original_datasets['poison_train'],
                original_datasets['poison_validation']
                ])
        data_idxes = np.array(all_dataset['idx'])
        sentences = np.array(all_dataset['sentence'])
        labels = np.array(all_dataset['label'])
        unique_labels = np.unique(labels)

        label_masks = {label: (data_idxes < poison_train_index) & (labels == label) for label in unique_labels}
        poison_mask = (data_idxes >= poison_train_index)
        
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
        
        activations = [torch.empty(0, model.config.hidden_size).cuda() for _ in range(0,19)]
            
        logger.info(f'{time.time() - begin_time} - Compute activations')

        with torch.no_grad():
            for data in tqdm(dataloader, desc="Compute activations", total=len(dataloader)):
                inputs = tokenizer(data['input'], return_tensors="pt", padding='longest').to('cuda')
                outputs = model(**inputs, output_hidden_states=True)

                for hidden_state in range(0, 19):
                    activations[hidden_state] = torch.cat((activations[hidden_state], outputs.hidden_states[hidden_state][:,-1,:]))
        for hidden_state in range(0, 19):
            activations[hidden_state] = activations[hidden_state].cpu().detach().numpy()

        biggest_diff_hidden_state = 0
        biggest_diff = -1
        biggest_diff_activation_pca = None
        logger.info(f'{time.time() - begin_time} - Compute mean difference of activations between labels')
        for hidden_state in tqdm(range(1, 19), desc="Compute mean difference of activations between labels at each hidden state", total=18):
            # Activation
            activation_original = activations[hidden_state]

            activation_normalized = StandardScaler().fit_transform(activation_original)

            activation_pca = PCA(n_components=15).fit_transform(activation_normalized)

            # activation_tsne = TSNE(n_components=2).fit_transform(activation_pca)

            # # Draw the activation with labels
            # fig = plt.figure(figsize=(20, 20))
            # activation_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            # for i in range(len(unique_labels)):
            #     label = unique_labels[i] 
            #     mask = label_masks[label]
            #     label_activation_tsne = activation_tsne[mask]
            #     plt.scatter(label_activation_tsne[:, 0], label_activation_tsne[:, 1], s=250, facecolors='none', linewidths=3, color=activation_colors[i], label=TaskPattern.get_labels(task_name, label))
            # poison_activation_tsne = activation_tsne[poison_mask]
            # plt.scatter(poison_activation_tsne[:, 0], poison_activation_tsne[:, 1], s=250, marker='x', linewidth=3, color='b', label='poison')
            # ax = plt.gca()
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            # ax.set_xticks([])
            # ax.set_yticks([])
            # plt.legend(fontsize=30, loc='upper right')
            # os.makedirs(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, f'hidden_state{hidden_state}'), exist_ok=True)
            # plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, f'hidden_state{hidden_state}', 'labels'))
            # plt.close()

            # metrics = dict()
            # metrics['total_variances'] = np.sum(np.var(activation_pca, axis=0, ddof=1))
            # metrics['poison_variances'] = np.sum(np.var(activation_pca[poison_mask], axis=0, ddof=1))
            # metrics['clean_variances'] = np.sum(np.var(activation_pca[~poison_mask], axis=0, ddof=1))
            # for label in unique_labels:
            #     mask = label_masks[label]
            #     metrics[f'{label}_variances'] = np.sum(np.var(activation_pca[mask], axis=0, ddof=1))

            # poison_mean = np.mean(activation_pca[poison_mask], axis=0)
            # metrics['poison_clean_diff'] = np.linalg.norm(poison_mean - np.mean(activation_pca[~poison_mask], axis=0), ord=2)
            # for i in range(len(unique_labels)):
            #     label = unique_labels[i]
            #     mask = label_masks[label]
            #     label_mean = np.mean(activation_pca[mask], axis=0)
            #     metrics[f'poison_{label}_diff'] = np.linalg.norm(poison_mean - label_mean, ord=2)

            diff = 0
            label_means = np.array([np.mean(activation_pca[label_masks[label]], axis=0) for label in unique_labels])
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    label_diff = np.linalg.norm(label_means[i] - label_means[j], ord=2)
                    diff += label_diff
            if diff >= biggest_diff:
                biggest_diff = diff
                biggest_diff_hidden_state = hidden_state
                biggest_diff_activation_pca = activation_pca

            # metrics['total_diff'] = diff
            # metrics_file = os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, f'hidden_state{hidden_state}', 'metrics.json')
            # os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            # with open(metrics_file, 'w') as f:
            #     metrics = {k: float(v) for k, v in metrics.items()}
            #     json.dump(metrics, f, indent=4)
            
        # poison_rates=[30, 20, 15]
        # for poison_rate in poison_rates:
        #     # distance mean
        #     mean_diff = np.zeros(0)
        #     for i, activation in enumerate(biggest_diff_activation_pca):
        #         label_distance = {label: np.mean(np.sqrt(np.sum((activation - biggest_diff_activation_pca[labels == label])**2, axis=1))) for label in unique_labels}
        #         closest_label = min(label_distance, key=label_distance.get)
        #         if closest_label == labels[i]:
        #             mean_diff = np.append(mean_diff, np.mean([label_distance[label] for label in unique_labels if label != labels[i]]))
        #         else:
        #             mean_diff = np.append(mean_diff, label_distance[closest_label] - label_distance[labels[i]])
            
        #     is_clean = (mean_diff >= 0) 
            
        #     mean_diff = np.zeros(0)
        #     for i, activation in enumerate(biggest_diff_activation_pca):
        #         label_distance = {label: np.mean(np.sqrt(np.sum((activation - biggest_diff_activation_pca[(labels == label) & (is_clean)])**2, axis=1))) for label in unique_labels}
        #         closest_label = min(label_distance, key=label_distance.get)
        #         if closest_label == labels[i]:
        #             mean_diff = np.append(mean_diff, np.mean([label_distance[label] for label in unique_labels if label != labels[i]]))
        #         else:
        #             mean_diff = np.append(mean_diff, label_distance[closest_label] - label_distance[labels[i]])

        #     negative_values = -mean_diff[mean_diff < 0]
        #     threshold = np.percentile(negative_values, 100 - poison_rate)
        #     is_poison = (mean_diff < 0) & (-mean_diff >= threshold)
            
        #     # Compute the tpr and fpr
        #     poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()
        #     poison_tpr = poison_tp / (poison_tp + poison_fn)
        #     poison_fpr = poison_fp / (poison_fp + poison_tn)
        #     clean_tn, clean_fp, clean_fn, clean_tp = confusion_matrix(~poison_mask, is_clean).ravel()
        #     clean_tpr = clean_tp / (clean_tp + clean_fn)
        #     clean_fpr = clean_fp / (clean_fp + clean_tn)

        #     metrics = dict()
        #     metrics['poison TPR'] = f'{poison_tpr}({poison_tp}/{poison_tp+poison_fn})'
        #     metrics['poison FPR'] = f'{poison_fpr}({poison_fp}/{poison_fp+poison_tn})'
        #     metrics['Clean TPR'] = f'{clean_tpr}({clean_tp}/{clean_tp+clean_fn})'
        #     metrics['Clean FPR'] = f'{clean_fpr}({clean_fp}/{clean_fp+clean_tn})'
            
        #     for hidden_state in range(1, 19):
        #         metrics_file = os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, f'hidden_state{hidden_state}', 'detect', f'hidden_state{biggest_diff_hidden_state}-{clean_rate}-{poison_rate}-{reclean_rate}-step0.json')
        #         os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        #         with open(metrics_file, 'w') as f:
        #             json.dump(metrics, f, indent=4)

        #     # activation_tsne = TSNE(n_components=2).fit_transform(biggest_diff_activation_pca)
        #     # fig = plt.figure(figsize=(20, 20))
        #     # poison_activation_tsne = activation_tsne[is_poison]
        #     # true_poison_activation_tsne = activation_tsne[poison_mask]
        #     # clean_activation_tsne = activation_tsne[is_clean]
        #     # plt.scatter(clean_activation_tsne[:, 0], clean_activation_tsne[:, 1], s=250, facecolors='none', linewidths=3, color='g', label='Clean')
        #     # plt.scatter(poison_activation_tsne[:, 0], poison_activation_tsne[:, 1], s=250, facecolors='none', linewidth=3, color='r', label='poison')
        #     # plt.scatter(true_poison_activation_tsne[:, 0], true_poison_activation_tsne[:, 1], s=250, marker='x', linewidth=3, color='b', label='True poison')
        #     # plt.legend()
        #     # plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, f'hidden_state{biggest_diff_hidden_state}', f'{clean_rate}-{poison_rate}-{reclean_rate}-poison'))
        #     # plt.close()
            
        #     logger.info(f'{time.time() - begin_time} - Detect poison')
        #     original_is_poison = is_poison
        #     original_is_clean = is_clean
        #     for hidden_state in tqdm(range(1, 19), desc="Detect poison in each hidden state", total=18):
        #         is_poison = original_is_poison.copy()
        #         is_clean = original_is_clean.copy()
        #         # Activation
        #         activation = activations[hidden_state]

        #         last_is_poison = is_poison
        #         lda_step = 1
        #         while lda_step <= self.lda_steps_limit:
        #             poison_activation = activation[is_poison]
        #             clean_activation = activation[is_clean]

        #             poison_clean = np.vstack([poison_activation, clean_activation])
        #             poison_clean_label = np.hstack([np.zeros(len(poison_activation)), np.ones(len(clean_activation))])
        #             activation_lda_origin = LDA().fit(poison_clean, poison_clean_label)

        #             activation_lda_prediction = activation_lda_origin.predict(activation)
        #             activation_lda_prediction = activation_lda_prediction == 0
        #             is_poison = activation_lda_prediction.copy()
                    
        #             # Compute the tpr and fpr
        #             poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()
        #             poison_tpr = poison_tp / (poison_tp + poison_fn)
        #             poison_fpr = poison_fp / (poison_fp + poison_tn)
        #             clean_tn, clean_fp, clean_fn, clean_tp = confusion_matrix(~poison_mask, ~is_poison).ravel()
        #             clean_tpr = clean_tp / (clean_tp + clean_fn)
        #             clean_fpr = clean_fp / (clean_fp + clean_tn)
        #             metrics = dict()
        #             metrics['poison TPR'] = f'{poison_tpr}({poison_tp}/{poison_tp+poison_fn})'
        #             metrics['poison FPR'] = f'{poison_fpr}({poison_fp}/{poison_fp+poison_tn})'
        #             metrics['Clean TPR'] = f'{clean_tpr}({clean_tp}/{clean_tp+clean_fn})'
        #             metrics['Clean FPR'] = f'{clean_fpr}({clean_fp}/{clean_fp+clean_tn})'
        #             metrics_file = os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, f'hidden_state{hidden_state}', 'detect', f'hidden_state{biggest_diff_hidden_state}-{clean_rate}-{poison_rate}-{reclean_rate}-step{lda_step}.json')
        #             os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        #             with open(metrics_file, 'w') as f:
        #                 json.dump(metrics, f, indent=4)
                    
        #             if np.all(is_poison == last_is_poison):
        #                 break
        #             last_is_poison = is_poison.copy()
                    
        #             lda_step += 1

        logger.info(f'{time.time() - begin_time} - Start filter the samples with wrong label')
        mean_diff = np.zeros(0)
        for i, activation in enumerate(biggest_diff_activation_pca):
            label_distance = {label: np.mean(np.sqrt(np.sum((activation - biggest_diff_activation_pca[labels == label])**2, axis=1))) for label in unique_labels}
            closest_label = min(label_distance, key=label_distance.get)
            if closest_label == labels[i]:
                mean_diff = np.append(mean_diff, np.mean([label_distance[label] for label in unique_labels if label != labels[i]]))
            else:
                mean_diff = np.append(mean_diff, label_distance[closest_label] - label_distance[labels[i]])
        
        is_clean = (mean_diff >= 0) 
        
        mean_diff = np.zeros(0)
        for i, activation in enumerate(biggest_diff_activation_pca):
            label_distance = {label: np.mean(np.sqrt(np.sum((activation - biggest_diff_activation_pca[(labels == label) & (is_clean)])**2, axis=1))) for label in unique_labels}
            closest_label = min(label_distance, key=label_distance.get)
            if closest_label == labels[i]:
                mean_diff = np.append(mean_diff, np.mean([label_distance[label] for label in unique_labels if label != labels[i]]))
            else:
                mean_diff = np.append(mean_diff, label_distance[closest_label] - label_distance[labels[i]])

        negative_values = -mean_diff[mean_diff < 0]
        threshold = np.percentile(negative_values, 100 - self.filter_rate * 100)
        is_poison = (mean_diff < 0) & (-mean_diff >= threshold)
        logger.info(f'{time.time() - begin_time} - Finish filter the samples with wrong label')
        
        activation = activations[18] # Use the last hidden state
        logger.info(f'{time.time() - begin_time} - Start detect poison')
        original_is_poison = is_poison
        original_is_clean = is_clean
        is_poison = original_is_poison.copy()
        is_clean = original_is_clean.copy()
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
            
            if np.all(is_poison == last_is_poison):
                break
            last_is_poison = is_poison.copy()
            
            lda_step += 1

        logger.info(f'{time.time() - begin_time} - Finish detect poison')

        def formatting_func(example):
            output_texts = []
            for i in range(len(example['sentence'])):
                text = TaskPattern.get_input(attacker_args['data']['task_name'], example['sentence'][i], example['label'][i])
                output_texts.append(text)
            return output_texts
        
        clean_train_clean_indices = np.where(~is_poison[(data_idxes >= clean_train_index) & (data_idxes < clean_validation_index)])[0]
        poison_train_clean_indices = np.where(~is_poison[(data_idxes >= poison_train_index) & (data_idxes < poison_validation_index)])[0]
        trainer = LogAsrTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=datasets.concatenate_datasets([
                original_datasets['clean_train'].select(clean_train_clean_indices),
                original_datasets['poison_train'].select(poison_train_clean_indices)
                ]),
            eval_dataset={
                'clean': original_datasets['clean_validation'], 
                'poison': original_datasets['poison_validation'], 
                'total': datasets.concatenate_datasets([original_datasets['clean_validation'], original_datasets['poison_validation']])
                },
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        logger.info(f'{time.time()-begin_time} - Start training')
        trainer.train()
        logger.info(f'{time.time()-begin_time} - Training finished')
        
        logger.info(f'{time.time()-begin_time} - Start evaluation')
        metrics = trainer.evaluate()
        logger.info(f'{time.time()-begin_time} - Evaluation finished')


        # Compute the tpr and fpr
        poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()
        poison_tpr = poison_tp / (poison_tp + poison_fn)
        poison_fpr = poison_fp / (poison_fp + poison_tn)

        return {
            'epoch': metrics['epoch'],
            'ASR': metrics['eval_poison_accuracy'],
            'ACC': metrics['eval_clean_accuracy'],
            'TPR': f'{poison_tpr}({poison_tp}/{poison_tp+poison_fn})',
            'FPR': f'{poison_fpr}({poison_fp}/{poison_fp+poison_tn})'
        }
