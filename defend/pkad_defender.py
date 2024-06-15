from .defender import Defender

import datasets
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn import config_context
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from scipy.stats import gaussian_kde
from matplotlib.font_manager import FontProperties

from log_asr_trainer import LogAsrTrainer
from task_pattern import TaskPattern

import time
import logging
from tqdm import tqdm
import os
import shutil
import json
import gc

logger = logging.getLogger("root")

class PkadDefender(Defender):    
    """
        Defender for PKAD
    
    Args:
        poison_rate (`float`, optional): The rate of poison data that defender expects to detect. Default to 0.15.
        lda_steps_limit (`int`, optional): The maximum number of steps for LDA. Default to 20.
        lda_batch_size (`int`, optional): The batch size of misclassified data for LDA. Default to 500.
        batch_size (`int`, optional): The batch size to compute hidden states. Default to 5.
    """
    def __init__(
            self,
            poison_rate=0.15,
            lda_steps_limit=20,
            batch_size=5,
            **kwargs
        ):

        super().__init__(**kwargs)

        self.poison_rate = poison_rate
        self.lda_steps_limit = lda_steps_limit
        self.batch_size = batch_size
        
    def defend(self, model, tokenizer, original_datasets, training_args, peft_config, attacker_args, model_args, begin_time):
        start_train = time.time()

        # clean_train_indices, poison_train_indices, poison_tn, poison_fp, poison_fn, poison_tp = self.detect(model, tokenizer, original_datasets, attacker_args['data']['task_name'], attacker_args['data']['poison_name'], attacker_args['train']['max_seq_length'], model_args['model_name_or_path'], begin_time)
        clean_train_indices, poison_train_indices , poison_tn, poison_fp, poison_fn, poison_tp = self.detect_with_log(model, tokenizer, original_datasets, attacker_args['data']['task_name'], attacker_args['data']['poison_name'], attacker_args['train']['max_seq_length'], model_args['model_name_or_path'], begin_time)

        # train_dataset = datasets.concatenate_datasets([
        #     original_datasets['clean_train'].select(clean_train_indices),
        #     original_datasets['poison_train'].select(poison_train_indices)
        # ])
        # eval_dataset = {
        #     'clean': original_datasets['clean_validation'],
        #     'poison': original_datasets['poison_validation']
        # }
        # self.train(model, tokenizer, train_dataset, eval_dataset, training_args, peft_config, attacker_args, begin_time)

        # end_train = time.time()
        
        # start_test, end_test, acc, asr = Defender.compute_acc_asr(model, tokenizer, original_datasets['clean_test'], original_datasets['poison_test'], attacker_args['data']['task_name'], training_args.per_device_eval_batch_size, attacker_args['train']['max_seq_length'], begin_time)

        return {
            'epoch': training_args.num_train_epochs,
            # 'ASR': asr,
            # 'ACC': acc,
            # 'ACC': Defender.compute_accuracy(model, tokenizer, datasets.concatenate_datasets([
            #                                                                 original_datasets['clean_train'],
            #                                                                 original_datasets['clean_validation'],
            #                                                                 original_datasets['clean_test'],
            #                                                                 ])
            #                                         , attacker_args['data']['task_name'], self.batch_size, attacker_args['train']['max_seq_length']),
            'TPR': f'{poison_tp / (poison_tp + poison_fn)}({poison_tp}/{poison_tp+poison_fn})',
            'FPR': f'{poison_fp / (poison_fp + poison_tn)}({poison_fp}/{poison_fp+poison_tn})',
            # 'train time': end_train - start_train,
            # 'test time': end_test - start_test
        } 

    def train(self, model, tokenizer, train_dataset, eval_dataset, training_args, peft_config, attacker_args, begin_time):
        def formatting_func(example):
            output_texts = []
            for i in range(len(example['sentence'])):
                text = TaskPattern.get_input(attacker_args['data']['task_name'], example['sentence'][i], example['label'][i])
                output_texts.append(text)
            return output_texts
        
        trainer = LogAsrTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            formatting_func=formatting_func,
            max_seq_length=attacker_args['train']['max_seq_length'],
        )

        logger.info(f'{time.time()-begin_time} - Start training')
        trainer.train()
        logger.info(f'{time.time()-begin_time} - Training finished')

    def detect(self, model, tokenizer, original_datasets, task_name, poison_name, max_length, model_name, begin_time):
        all_dataset = datasets.concatenate_datasets([
                original_datasets['clean_train'],
                original_datasets['clean_validation'],
                original_datasets['clean_test'],
                original_datasets['poison_train'],
                original_datasets['poison_validation'],
                original_datasets['poison_test'],
                ])
        clean_train_begin = 0
        clean_eval_begin = clean_train_begin + len(original_datasets['clean_train'])
        clean_test_begin = clean_eval_begin + len(original_datasets['clean_validation'])
        poison_train_begin = clean_test_begin + len(original_datasets['clean_test'])

        labels = np.array(all_dataset['label'])
        unique_labels = np.unique(labels)

        poison_mask = np.array([False if i < poison_train_begin else True for i in range(len(all_dataset))])
        label_masks = {label: ~poison_mask & (labels == label) for label in unique_labels}
        
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
            batch_size=self.batch_size
            )
        
        # Initial Partitioning Strategy Based on the Dirty Label of Poisoned Data
        activations = [torch.empty((len(all_dataset), model.config.hidden_size)).cuda() for _ in range(0,model.config.num_hidden_layers + 1)]
        with torch.no_grad():
            for batch_num, data in tqdm(enumerate(dataloader), desc="Compute hidden states", total=len(dataloader)):
                inputs = tokenizer(data['input'], return_tensors="pt", padding='longest', truncation=True, max_length=max_length).to('cuda')
                outputs = model(**inputs, output_hidden_states=True)
                for hidden_state in range(0, model.config.num_hidden_layers + 1):
                    activations[hidden_state][batch_num * self.batch_size: (batch_num + 1) * self.batch_size] = outputs.hidden_states[hidden_state][:,-1,:]

        # Calculate the optimal layer for the first stage
        optimal_hidden_state = 0
        biggest_distan = -1
        optimal_activation_pca = None
        logger.info(f'{time.time() - begin_time} - Compute distance of hidden states between labels')
        for hidden_state in tqdm(range(1, model.config.num_hidden_layers + 1), desc="Compute distance of hidden states between labels of each layer", total=model.config.num_hidden_layers):
            # Activation
            activation_original = activations[hidden_state].cpu().detach().numpy()

            activation_normalized = StandardScaler().fit_transform(activation_original)

            activation_pca = PCA(n_components=10).fit_transform(activation_normalized)
            activation_pca = torch.tensor(activation_pca).cuda()

            label_means = [torch.mean(activation_pca[label_masks[label]], axis=0) for label in unique_labels]
            label_covariances = [torch.inverse(torch.cov(activation_pca[label_masks[label]].t())) for label in unique_labels]
            total_covatiance = torch.inverse(torch.cov(activation_pca.t()))
            intra_distan = 0
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    mean_diff = label_means[i] - label_means[j]
                    intra_distan += torch.sqrt((mean_diff @ total_covatiance @ mean_diff.t()))

            if intra_distan >= biggest_distan:
                biggest_distan = intra_distan
                optimal_hidden_state = hidden_state
                optimal_activation_pca = activation_pca

        logger.info(f'{time.time() - begin_time} - The layer of biggest distance of the first stage is {optimal_hidden_state}')

        logger.info(f'{time.time() - begin_time} - Start filter the samples with wrong label')

        # Calculate the misclassified dataset
        # Calculate the degree of misclassification
        
        # Mahalanobis Distance
        mahalanobis_distan = torch.zeros(0)
        label_means = [torch.mean(optimal_activation_pca[label_masks[label]], axis=0) for label in unique_labels]
        label_covariances = [torch.inverse(torch.cov(optimal_activation_pca[label_masks[label]].t())) for label in unique_labels]
        for i in range(len(optimal_activation_pca)):
            activation_label_distances = {label: torch.sqrt((optimal_activation_pca[i] - label_mean) @ label_covariance @ (optimal_activation_pca[i] - label_mean).t()) for label, label_mean, label_covariance in zip(unique_labels, label_means, label_covariances)}
            closest_label = min(activation_label_distances, key=activation_label_distances.get)
            if closest_label == labels[i]:
                mahalanobis_distan = torch.cat((mahalanobis_distan, torch.tensor([1])))
            else:
                mahalanobis_distan = torch.cat((mahalanobis_distan, torch.tensor([activation_label_distances[closest_label] - activation_label_distances[labels[i]]])))
        
        is_clean = (mahalanobis_distan >= 0)

        negative_values = -mahalanobis_distan[mahalanobis_distan < 0]
        # Calculate the preliminary poisoned data marking ratio
        misclassify_rate = torch.sum((mahalanobis_distan < 0)) / len(mahalanobis_distan)
        if misclassify_rate <= self.poison_rate:
            filter_rate = 1
        else:
            filter_rate = (self.poison_rate / (misclassify_rate - self.poison_rate)) ** (1.33)
            filter_rate = min([filter_rate, 1])
        threshold = torch.quantile(negative_values, 1 - filter_rate)
        is_poison = (mahalanobis_distan < 0) & (-mahalanobis_distan >= threshold)

        logger.info(f'{time.time() - begin_time} - Finish filter the samples with wrong label')

        # Iterative Partitioning Strategy Based on the Shared Trigger of Poisoned Data

        # Calculate the optimal layer for the second stage
        optimal_hidden_state = 0
        biggest_distan = -1
        optimal_activation = None
        logger.info(f'{time.time() - begin_time} - Start detect poison')
        for hidden_state in tqdm(range(1, model.config.num_hidden_layers + 1), desc="Compute distance ratio of embeddings by lda between the data labeled as clean and the data labeled as poisoned of each layer", total=model.config.num_hidden_layers):
            activation = activations[hidden_state]

            poison_activation = activation[is_poison]
            clean_activation = activation[is_clean]

            poison_clean = torch.concat((poison_activation, clean_activation), dim=0)
            poison_clean_label = torch.concat((torch.zeros(len(poison_activation)), torch.ones(len(clean_activation)))).cuda()
            with config_context(array_api_dispatch=True):
                activation_lda = LDA().fit_transform(poison_clean, poison_clean_label)

            poison_mean = torch.mean(activation_lda[poison_clean_label == 0], dim=0)
            clean_mean = torch.mean(activation_lda[poison_clean_label == 1], dim=0)
            total_variance = torch.var(activation_lda, dim=0, unbiased=True)
            intra_distan = torch.abs(poison_mean - clean_mean) /  torch.sqrt(total_variance)

            if intra_distan  >= biggest_distan:
                biggest_distan = intra_distan
                optimal_hidden_state = hidden_state
                optimal_activation = activation

        logger.info(f'{time.time() - begin_time} - The layer of biggest distance of the second stage is {optimal_hidden_state}')

        #  Iteratively update the poisoned dataset
        random_indices = torch.randperm(len(is_poison))
        lda_step = 1
        last_is_poison = is_poison
        while lda_step <= self.lda_steps_limit:
            poison_activation = optimal_activation[random_indices][is_poison[random_indices]]
            clean_activation = optimal_activation[random_indices][is_clean[random_indices]]

            poison_clean = torch.concat([poison_activation, clean_activation], dim=0)
            poison_clean_label = torch.concat([torch.zeros(len(poison_activation)), torch.ones(len(clean_activation))]).cuda()
            with config_context(array_api_dispatch=True):
                activation_lda_origin = LDA().fit(poison_clean, poison_clean_label)
                activation_lda_prediction = activation_lda_origin.predict(optimal_activation)

            activation_lda_prediction = activation_lda_prediction == 0
            is_poison[random_indices] = activation_lda_prediction[random_indices].cpu()

            poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison.cpu().detach().numpy()).ravel()
            
            if torch.all(is_poison == last_is_poison):
                break
            last_is_poison = is_poison.clone()
            
            lda_step += 1

        # Compute the tpr and fpr
        poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()

        logger.info(f'{time.time() - begin_time} - Finish detect poison\nTPR: {poison_tp / (poison_tp + poison_fn)}({poison_tp}/{poison_tp+poison_fn})\nFPR: {poison_fp / (poison_fp + poison_tn)}({poison_fp}/{poison_fp+poison_tn})')

        clean_train_indices = np.where(~is_poison[clean_train_begin:clean_train_begin + len(original_datasets['clean_train'])])[0]
        poison_train_indices = np.where(~is_poison[poison_train_begin:poison_train_begin + len(original_datasets['poison_train'])])[0]

        return clean_train_indices, poison_train_indices, poison_tn, poison_fp, poison_fn, poison_tp
 
    def detect_with_log(self, model, tokenizer, original_datasets, task_name, poison_name, max_length, model_name, begin_time):
        
        font_prop = FontProperties(fname='/home/chenyu/.Ascian/.program/anaconda/envs/moderate_env/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/TimesNewRomanPSBoldMT.ttf', size=35)
        if os.path.exists(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name)):
            shutil.rmtree(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name))
        
        all_dataset = datasets.concatenate_datasets([
                original_datasets['clean_train'],
                original_datasets['clean_validation'],
                original_datasets['clean_test'],
                original_datasets['poison_train'],
                original_datasets['poison_validation'],
                original_datasets['poison_test'],
                ])
        clean_train_begin = 0
        clean_eval_begin = clean_train_begin + len(original_datasets['clean_train'])
        clean_test_begin = clean_eval_begin + len(original_datasets['clean_validation'])
        poison_train_begin = clean_test_begin + len(original_datasets['clean_test'])

        labels = np.array(all_dataset['label'])
        unique_labels = np.unique(labels)

        poison_mask = np.array([False if i < poison_train_begin else True for i in range(len(all_dataset))])
        label_masks = {label: ~poison_mask & (labels == label) for label in unique_labels}
        
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
            batch_size=self.batch_size
            )
        
        random_indices = torch.randperm(len(all_dataset))
        if len(all_dataset) >= 10000:
            selected_indices = random_indices[0: 10000]
        else:
            selected_indices = random_indices
            
        # Initial Partitioning Strategy Based on the Dirty Label of Poisoned Data

        activations = [torch.empty((len(all_dataset), model.config.hidden_size)).cuda() for _ in range(0,model.config.num_hidden_layers + 1)]
        with torch.no_grad():
            for batch_num, data in tqdm(enumerate(dataloader), desc="Compute hidden states", total=len(dataloader)):
                inputs = tokenizer(data['input'], return_tensors="pt", padding='longest', truncation=True, max_length=max_length).to('cuda')
                outputs = model(**inputs, output_hidden_states=True)

                for hidden_state in range(0, model.config.num_hidden_layers + 1):
                    activations[hidden_state][batch_num * self.batch_size: (batch_num + 1) * self.batch_size] = outputs.hidden_states[hidden_state][:,-1,:]

        # Calculate the optimal layer for the first stage
        optimal_hidden_state = 0
        biggest_distan = -1
        optimal_activation_pca = None
        optimal_activation_tsne = None
        logger.info(f'{time.time() - begin_time} - Compute distance of hidden states between labels')
        for hidden_state in tqdm(range(1, model.config.num_hidden_layers + 1), desc="Compute distance of hidden states between labels of each layer", total=model.config.num_hidden_layers):
            # Activation
            activation_original = activations[hidden_state].cpu().detach().numpy()

            activation_normalized = StandardScaler().fit_transform(activation_original)

            activation_pca = PCA(n_components=10).fit_transform(activation_normalized)

            # Draw the activation with labels
            if task_name == 'agnews':
                activation_tsne = TSNE(n_components=2).fit_transform(activation_pca[selected_indices])
                plt.figure(figsize=(10, 10))
                activation_colors = ['#82b0d2', '#8ecfc9', '#96c37d', '#f3d266', ]
                for i in range(len(unique_labels)):
                    label = unique_labels[i] 
                    mask = label_masks[label][selected_indices]
                    label_activation_tsne = activation_tsne[mask]
                    plt.scatter(label_activation_tsne[:, 0], label_activation_tsne[:, 1], color=activation_colors[i], label=TaskPattern.get_labels(task_name, label).strip().capitalize())
                poison_activation_tsne = activation_tsne[poison_mask[selected_indices]]
                ax = plt.gca()
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                os.makedirs(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'hidden_state{hidden_state}'), exist_ok=True)
                plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'hidden_state{hidden_state}', 'labels_wo_poison'))
                plt.close()

                plt.figure(figsize=(10, 10))
                activation_colors = ['#82b0d2', '#8ecfc9', '#96c37d', '#f3d266', ]
                for i in range(len(unique_labels)):
                    label = unique_labels[i] 
                    mask = label_masks[label][selected_indices]
                    label_activation_tsne = activation_tsne[mask]
                    plt.scatter(label_activation_tsne[:, 0], label_activation_tsne[:, 1], color=activation_colors[i], label=TaskPattern.get_labels(task_name, label).strip().capitalize())
                poison_activation_tsne = activation_tsne[poison_mask[selected_indices]]
                plt.scatter(poison_activation_tsne[:, 0], poison_activation_tsne[:, 1],  marker='x', color='#d8383a', label='Poison', s=100, linewidths=3)
                ax = plt.gca()
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                os.makedirs(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'hidden_state{hidden_state}'), exist_ok=True)
                plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'hidden_state{hidden_state}', 'labels_w_poison'))
                plt.close()

            activation_pca = torch.tensor(activation_pca).cuda()

            label_means = [torch.mean(activation_pca[label_masks[label]], axis=0) for label in unique_labels]
            label_covariances = [torch.inverse(torch.cov(activation_pca[label_masks[label]].t())) for label in unique_labels]
            total_covatiance = torch.inverse(torch.cov(activation_pca.t()))
            intra_distan = 0
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    mean_diff = label_means[i] - label_means[j]
                    intra_distan += torch.sqrt((mean_diff @ total_covatiance @ mean_diff.t()))

            if intra_distan >= biggest_distan:
                biggest_distan = intra_distan
                optimal_hidden_state = hidden_state
                optimal_activation_pca = activation_pca
                if task_name == 'agnews':
                    optimal_activation_tsne = activation_tsne

        if task_name == 'agnews':
            plt.figure(figsize=(20, 2))
            activation_colors = ['#82b0d2', '#8ecfc9', '#96c37d', '#f3d266', ]
            for i in range(len(unique_labels)):
                label = unique_labels[i] 
                mask = label_masks[label][selected_indices]
                label_activation_tsne = activation_tsne[mask]
                plt.scatter([], [], color=activation_colors[i], label=TaskPattern.get_labels(task_name, label).strip().capitalize(), s=200)
            plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.05), prop=font_prop, borderpad=0.2)
            plt.axis('off')  
            os.makedirs(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name), exist_ok=True)
            plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'first_stage_optimal{optimal_hidden_state}_legend_wo_poison'))
            plt.close()

            plt.figure(figsize=(25, 2))
            activation_colors = ['#82b0d2', '#8ecfc9', '#96c37d', '#f3d266', ]
            for i in range(len(unique_labels)):
                if i == 1:
                    plt.scatter([], [],  marker='x', color='#d8383a', label='Poison (World)', s=200, linewidths=3)
                if i == 2:
                    plt.hist([], bins=10, density=False, alpha=0.5, color='g', label='Misclassified Clean')
                if i == 3:
                    plt.hist([], bins=10, density=False, alpha=0.5, color='#d8383a', label='Misclassified Poison')
                label = unique_labels[i] 
                mask = label_masks[label][selected_indices]
                label_activation_tsne = activation_tsne[mask]
                plt.scatter([], [], color=activation_colors[i], label=TaskPattern.get_labels(task_name, label).strip().capitalize(), s=200)
            plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.05), prop=font_prop, borderpad=0.2)
            plt.axis('off')  
            os.makedirs(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name), exist_ok=True)
            plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'first_stage_optimal{optimal_hidden_state}_legend_w_poison'))
            plt.close()

        logger.info(f'{time.time() - begin_time} - The layer of biggest distance of the first stage is {optimal_hidden_state}')

        logger.info(f'{time.time() - begin_time} - Start filter the samples with wrong label')

        # Calculate the misclassified dataset
        # Calculate the degree of misclassification
        
        # Mahalanobis Distance
        mahalanobis_distan = torch.zeros(0)
        label_means = [torch.mean(optimal_activation_pca[label_masks[label]], axis=0) for label in unique_labels]
        label_covariances = [torch.inverse(torch.cov(optimal_activation_pca[label_masks[label]].t())) for label in unique_labels]
        for i in range(len(optimal_activation_pca)):
            activation_label_distances = {label: torch.sqrt((optimal_activation_pca[i] - label_mean) @ label_covariance @ (optimal_activation_pca[i] - label_mean).t()) for label, label_mean, label_covariance in zip(unique_labels, label_means, label_covariances)}
            closest_label = min(activation_label_distances, key=activation_label_distances.get)
            if closest_label == labels[i]:
                mahalanobis_distan = torch.cat((mahalanobis_distan, torch.tensor([1])))
            else:
                mahalanobis_distan = torch.cat((mahalanobis_distan, torch.tensor([activation_label_distances[closest_label] - activation_label_distances[labels[i]]])))
        is_clean = (mahalanobis_distan >= 0)
        negative_values = -mahalanobis_distan[mahalanobis_distan < 0]

        # Draw the activation of misclassified samples
        if task_name == 'agnews':
            data1 = -mahalanobis_distan[selected_indices][(mahalanobis_distan < 0)[selected_indices]][(~poison_mask)[selected_indices][(mahalanobis_distan < 0)[selected_indices]]]
            data2 = -mahalanobis_distan[selected_indices][(mahalanobis_distan < 0)[selected_indices]][(poison_mask)[selected_indices][(mahalanobis_distan < 0)[selected_indices]]]
            xmin = min(torch.min(data1), torch.min(data2))
            xmax = max(torch.max(data1), torch.max(data2))

            num_bins = 20
            bins = np.linspace(xmin, xmax, num_bins+1)

            fig, ax = plt.subplots(figsize=(20, 9))

            plt.hist(data1, bins=bins, density=True, alpha=0.5, color='g', label='Misclassified Clean')
            plt.hist(data2, bins=bins, density=True, alpha=0.5, color='#d8383a', label='Misclassified Poison')
            ax.set_xlabel('m(x)',  fontproperties=font_prop)
            plt.setp(plt.gca().get_xticklabels(), fontproperties=font_prop)

            ax.set_ylabel('Density',  fontproperties=font_prop)
            plt.setp(plt.gca().get_yticklabels(), fontproperties=font_prop)
            plt.legend(loc='upper right', prop=font_prop, borderpad=0.2)

            plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'first_stage_optimal{optimal_hidden_state}-misclassified'))
            plt.close()

        if poison_name == 'badnets':
            metrics = {
                'mahalanobis': (torch.sum((mahalanobis_distan >= 0)[~poison_mask]) / len(mahalanobis_distan[~poison_mask])).item(),
                'model': Defender.compute_accuracy(model, tokenizer, datasets.concatenate_datasets([
                                                                            original_datasets['clean_train'],
                                                                            original_datasets['clean_validation'],
                                                                            original_datasets['clean_test'],
                                                                            ])
                                                    , task_name, self.batch_size, max_length)
            }
            metrics_file = os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'first_stage_optimal{optimal_hidden_state}-acc.json')
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)

        # Calculate the preliminary poisoned data marking ratio
        misclassify_rate = torch.sum((mahalanobis_distan < 0)) / len(mahalanobis_distan)
        if misclassify_rate <= self.poison_rate:
            filter_rate = 1
        else:
            filter_rate = (self.poison_rate / (misclassify_rate - self.poison_rate)) ** (1.33)
            filter_rate = min([filter_rate, 1])
        threshold = torch.quantile(negative_values, 1 - filter_rate)
        is_poison = (mahalanobis_distan < 0) & (-mahalanobis_distan >= threshold)
        
        logger.info(f'{time.time() - begin_time} - Finish filter the samples with wrong label')

        # Iterative Partitioning Strategy Based on the Shared Trigger of Poisoned Data

        # Calculate the optimal layer for the second stage
        optimal_hidden_state = 0
        biggest_distan = -1
        distans = []
        optimal_activation = None
        logger.info(f'{time.time() - begin_time} - Start detect poison')
        for hidden_state in tqdm(range(1, model.config.num_hidden_layers + 1), desc="Compute distance of embeddings by lda between the data labeled as clean and the data labeled as poisoned of each layer", total=model.config.num_hidden_layers):
            activation = activations[hidden_state]

            poison_activation = activation[is_poison]
            clean_activation = activation[is_clean]

            poison_clean = torch.concat((poison_activation, clean_activation), dim=0)
            poison_clean_label = torch.concat((torch.zeros(len(poison_activation)), torch.ones(len(clean_activation)))).cuda()
            with config_context(array_api_dispatch=True):
                activation_lda = LDA().fit_transform(poison_clean, poison_clean_label)

            poison_mean = torch.mean(activation_lda[poison_clean_label == 0], dim=0)
            clean_mean = torch.mean(activation_lda[poison_clean_label == 1], dim=0)
            total_variance = torch.var(activation_lda, dim=0, unbiased=True)
            intra_distan = torch.abs(poison_mean - clean_mean) /  torch.sqrt(total_variance)

            if intra_distan  >= biggest_distan:
                biggest_distan = intra_distan
                optimal_hidden_state = hidden_state
                optimal_activation = activation
            distans.append(intra_distan)

        logger.info(f'{time.time() - begin_time} - The layer of biggest distance of the second stage is {optimal_hidden_state}')

        # rsme of different functions
        poison_misclassify = torch.sum((mahalanobis_distan < 0) & poison_mask)
        functions = [
            lambda x: poison_misclassify * (x ** (1 /2)),
            lambda x: poison_misclassify * (x ** (1 /3)),
            lambda x: poison_misclassify * (x ** (1 /4)),
            lambda x: poison_misclassify * (x ** (1 /5)),
            lambda x: poison_misclassify / np.log2(2) * np.log2(1 + x),
            lambda x: poison_misclassify / np.log2(4) * np.log2(2* (1 + x)),
            lambda x: poison_misclassify / np.log2(8) * np.log2(4* (1 + x)),
            lambda x: poison_misclassify / np.log2(16) * np.log2(8* (1 + x)),
        ]
        functions_rsme = [0 for _ in range(len(functions))]
        filter_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        poison_tps = []
        poison_fps = []
        for filter_rate in filter_rates:
            threshold = torch.quantile(negative_values, 1 - filter_rate)
            is_poison = (mahalanobis_distan < 0) & (-mahalanobis_distan >= threshold)
            # Compute the tpr and fpr
            poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison.cpu().detach().numpy()).ravel()
            poison_tps.append(int(poison_tp))
            poison_fps.append(int(poison_fp))
            functions_rsme = [rsme + (poison_tp - function(filter_rate)) ** 2 for rsme, function in zip(functions_rsme, functions)]
        functions_rsme = [(rsme / len(filter_rates)).item() ** (1/2) for rsme in functions_rsme]
        rsme_file = os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'second_stage_optimal{optimal_hidden_state}-rsme.json')
        os.makedirs(os.path.dirname(rsme_file), exist_ok=True)
        with open(rsme_file, 'w') as f:
            json.dump( functions_rsme, f, indent=4)

        rates_tps_fps = {
            'rates': filter_rates,
            'tps': poison_tps,
            'fps': poison_fps
        }
        rates_tps_fps_file = os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'second_stage_optimal{optimal_hidden_state}-rates_tps_fps.json')
        os.makedirs(os.path.dirname(rates_tps_fps_file), exist_ok=True)
        with open(rates_tps_fps_file, 'w') as f:
            json.dump(rates_tps_fps, f, indent=4)
        

        if task_name == 'agnews':
            # The values of the hidden state after LDA linear transformation
            poison_activation = optimal_activation[selected_indices][poison_mask[selected_indices]]
            clean_activation = optimal_activation[selected_indices][~poison_mask[selected_indices]]
            poison_clean = torch.concat([poison_activation, clean_activation], dim=0)
            poison_clean_label = torch.concat([torch.zeros(len(poison_activation)), torch.ones(len(clean_activation))]).cuda()
            with config_context(array_api_dispatch=True):
                activation_lda = LDA().fit_transform(poison_clean, poison_clean_label)
            jitter = 0.05 * torch.rand(len(activation_lda), 1) - 0.025
            plt.figure(figsize=(20, 7))
            plt.scatter(activation_lda[:len(poison_activation)].cpu(), jitter[:len(poison_activation)].cpu(), s=200, linewidth=3, marker='x', color='#d8383a', label='Poison')
            plt.scatter(activation_lda[len(poison_activation):].cpu(), jitter[len(poison_activation):].cpu(), s=200, color='#54b345', label='Clean')
            ax = plt.gca()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.legend(loc='upper right', prop=font_prop, borderpad=0.2)
            plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'second_stage_optimal{optimal_hidden_state}-lda'))
            plt.close()


        #  Iteratively update the poisoned dataset
        misclassify_rate = torch.sum((mahalanobis_distan < 0)) / len(mahalanobis_distan)
        if misclassify_rate <= self.poison_rate:
            filter_rate = 1
        else:
            filter_rate = (self.poison_rate / (misclassify_rate - self.poison_rate)) ** (1.33)
            filter_rate = min([filter_rate, 1])
        threshold = torch.quantile(negative_values, 1 - filter_rate)
        is_poison = (mahalanobis_distan < 0) & (-mahalanobis_distan >= threshold)
        poison_tps = []
        poison_fps = []
        poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()
        poison_tps.append(int(poison_tp))
        poison_fps.append(int(poison_fp))
        random_indices = torch.randperm(len(is_poison))
        lda_step = 1
        last_is_poison = is_poison
        while lda_step <= self.lda_steps_limit:
            poison_activation = optimal_activation[random_indices][is_poison[random_indices]]
            clean_activation = optimal_activation[random_indices][is_clean[random_indices]]

            poison_clean = torch.concat([poison_activation, clean_activation], dim=0)
            poison_clean_label = torch.concat([torch.zeros(len(poison_activation)), torch.ones(len(clean_activation))]).cuda()
            with config_context(array_api_dispatch=True):
                activation_lda_origin = LDA().fit(poison_clean, poison_clean_label)
                activation_lda_prediction = activation_lda_origin.predict(optimal_activation)

            activation_lda_prediction = activation_lda_prediction == 0
            is_poison[random_indices] = activation_lda_prediction[random_indices].cpu()

            poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison.cpu().detach().numpy()).ravel()
            poison_tps.append(int(poison_tp))
            poison_fps.append(int(poison_fp))
            
            if torch.all(is_poison == last_is_poison):
                break
            last_is_poison = is_poison.clone()
            
            lda_step += 1

        tps_fps = {
            'tps': poison_tps,
            'fps': poison_fps
        }
        tps_fps_file = os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'second_stage_optimal{optimal_hidden_state}-tps_fps.json')
        os.makedirs(os.path.dirname(tps_fps_file), exist_ok=True)
        with open(tps_fps_file, 'w') as f:
            json.dump(tps_fps, f, indent=4)

        # Compute the tpr and fpr
        poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison.cpu().detach().numpy()).ravel()

        logger.info(f'{time.time() - begin_time} - Finish detect poison\nTPR: {poison_tp / (poison_tp + poison_fn)}({poison_tp}/{poison_tp+poison_fn})\nFPR: {poison_fp / (poison_fp + poison_tn)}({poison_fp}/{poison_fp+poison_tn})')

        clean_train_indices = torch.where(~is_poison[clean_train_begin:clean_train_begin + len(original_datasets['clean_train'])])[0]
        poison_train_indices = torch.where(~is_poison[poison_train_begin:poison_train_begin + len(original_datasets['poison_train'])])[0]

        return clean_train_indices, poison_train_indices, poison_tn, poison_fp, poison_fn, poison_tp
 