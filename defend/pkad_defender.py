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
            lda_batch_size=500,
            batch_size=5,
            **kwargs
        ):

        super().__init__(**kwargs)

        self.poison_rate = poison_rate
        self.lda_steps_limit = lda_steps_limit
        self.lda_batch_size = lda_batch_size
        self.batch_size = batch_size
        
    def defend(self, model, tokenizer, original_datasets, training_args, peft_config, attacker_args, model_args, begin_time):
        start_train = time.time()

        # clean_train_indices, poison_train_indices, poison_tn, poison_fp, poison_fn, poison_tp = self.detect(model, tokenizer, original_datasets, attacker_args['data']['task_name'], attacker_args['data']['poison_name'], attacker_args['train']['max_seq_length'], model_args['model_name_or_path'], begin_time)
        clean_train_indices, poison_train_indices , poison_tn, poison_fp, poison_fn, poison_tp = self.detect_with_log(model, tokenizer, original_datasets, attacker_args['data']['task_name'], attacker_args['data']['poison_name'], attacker_args['train']['max_seq_length'], model_args['model_name_or_path'], begin_time)

        train_dataset = datasets.concatenate_datasets([
            original_datasets['clean_train'].select(clean_train_indices),
            original_datasets['poison_train'].select(poison_train_indices)
        ])
        eval_dataset = {
            'clean': original_datasets['clean_validation'],
            'poison': original_datasets['poison_validation']
        }
        self.train(model, tokenizer, train_dataset, eval_dataset, training_args, peft_config, attacker_args, begin_time)

        end_train = time.time()
        
        start_test, end_test, acc, asr = Defender.compute_acc_asr(model, tokenizer, original_datasets['clean_test'], original_datasets['poison_test'], attacker_args['data']['task_name'], training_args.per_device_eval_batch_size, attacker_args['train']['max_seq_length'], begin_time)

        return {
            'epoch': training_args.num_train_epochs,
            'ASR': asr,
            'ACC': acc,
            'TPR': f'{poison_tp / (poison_tp + poison_fn)}({poison_tp}/{poison_tp+poison_fn})',
            'FPR': f'{poison_fp / (poison_fp + poison_tn)}({poison_fp}/{poison_fp+poison_tn})',
            'train time': end_train - start_train,
            'test time': end_test - start_test
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
        biggest_distan_ratio = -1
        optimal_activation_pca = None
        logger.info(f'{time.time() - begin_time} - Compute distance ratio of hidden states between labels')
        for hidden_state in tqdm(range(1, model.config.num_hidden_layers + 1), desc="Compute distance ratio of hidden states between labels of each layer", total=model.config.num_hidden_layers):
            # Activation
            activation_original = activations[hidden_state].cpu().detach().numpy()

            activation_normalized = StandardScaler().fit_transform(activation_original)

            activation_pca = PCA(n_components=10).fit_transform(activation_normalized)

            # Euclidean Distance
            label_means = [np.mean(activation_pca[label_masks[label]], axis=0) for label in unique_labels]
            inner_distance = 0
            for label, label_mean in zip(unique_labels, label_means):
                inner_distance += np.mean(np.linalg.norm(activation_pca[label_masks[label]] - label_mean, axis=1))
            outer_distance = 0
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    outer_distance += np.linalg.norm(label_means[i] - label_means[j], ord=2)
            distan_ratio = outer_distance / inner_distance

            if distan_ratio >= biggest_distan_ratio:
                biggest_distan_ratio = distan_ratio
                optimal_hidden_state = hidden_state
                optimal_activation_pca = activation_pca


        logger.info(f'{time.time() - begin_time} - The layer of biggest distance ratio of the first stage is {optimal_hidden_state}')

        logger.info(f'{time.time() - begin_time} - Start filter the samples with wrong label')

        # Calculate the misclassified dataset
        # Calculate the degree of misclassification
        
        # Mahalanobis Distance
        mahalanobis_distan = np.zeros(0)
        label_means = np.array([np.mean(optimal_activation_pca[label_masks[label]], axis=0) for label in unique_labels])
        label_covariances = np.array([np.cov(optimal_activation_pca[label_masks[label]], rowvar=False) for label in unique_labels])
        for i, activation in enumerate(optimal_activation_pca):
            label_distances = {label: np.sqrt((activation - label_mean).T @ np.linalg.inv(label_covariance) @ (activation - label_mean)) for label, label_mean, label_covariance in zip(unique_labels, label_means, label_covariances)}
            closest_label = min(label_distances, key=label_distances.get)
            if closest_label == labels[i]:
                mahalanobis_distan = np.append(mahalanobis_distan, 1)
            else:
                mahalanobis_distan = np.append(mahalanobis_distan, label_distances[closest_label] - label_distances[labels[i]])
        
        is_clean = (mahalanobis_distan >= 0)

        negative_values = -mahalanobis_distan[mahalanobis_distan < 0]
        # Calculate the preliminary poisoned data marking ratio
        misclassify_rate = np.sum((mahalanobis_distan < 0)) / len(mahalanobis_distan)
        filter_rate = (self.poison_rate / (misclassify_rate - self.poison_rate)) ** (1.33)
        filter_rate = min([filter_rate, 1])
        threshold = np.percentile(negative_values, 100 - filter_rate * 100)
        is_poison = (mahalanobis_distan < 0) & (-mahalanobis_distan >= threshold)

        logger.info(f'{time.time() - begin_time} - Finish filter the samples with wrong label')

        # Iterative Partitioning Strategy Based on the Shared Trigger of Poisoned Data

        # Calculate the optimal layer for the second stage
        optimal_hidden_state = 0
        biggest_distan_ratio = -1
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
            poison_inner_distance = torch.mean(torch.norm(activation_lda[poison_clean_label == 0] - torch.mean(activation_lda[poison_clean_label == 0], dim=0), dim=1))
            clean_inner_distance = torch.mean(torch.norm(activation_lda[poison_clean_label == 1] - torch.mean(activation_lda[poison_clean_label == 1], dim=0), dim=1))
            outer_distance = torch.norm(poison_mean - clean_mean, p=2)
            distan_ratio = outer_distance / (poison_inner_distance + clean_inner_distance)

            if distan_ratio >= biggest_distan_ratio:
                biggest_distan_ratio = distan_ratio
                optimal_hidden_state = hidden_state
                optimal_activation = activation

        logger.info(f'{time.time() - begin_time} - The layer of biggest distance ratio of the second stage is {optimal_hidden_state}')

        #  Iteratively update the poisoned dataset
        # We found that for datasets with a very large number of samples, processing misclassified data in batches yields better results.
        clean_activation = optimal_activation[is_clean]
        is_poison = np.array([False for _ in range(len(optimal_activation))])
        random_indices = torch.randperm(len(negative_values))
        batch_begin = 0
        while batch_begin + 2 * self.lda_batch_size < len(negative_values):
            negative_values_batch = negative_values[random_indices[batch_begin: batch_begin+self.lda_batch_size]]
            threshold_batch = np.percentile(negative_values_batch, 100 - filter_rate * 100)
            is_poison_batch = (negative_values_batch >= threshold_batch)
            
            last_is_poison = is_poison
            lda_step = 1
            while lda_step <= self.lda_steps_limit:
                if lda_step == 1:
                    poison_activation = optimal_activation[mahalanobis_distan < 0][random_indices[batch_begin: batch_begin+self.lda_batch_size]][is_poison_batch]
                else: 
                    poison_activation = optimal_activation[mahalanobis_distan < 0][random_indices[batch_begin: batch_begin+self.lda_batch_size]][is_poison[mahalanobis_distan < 0][random_indices[batch_begin: batch_begin+self.lda_batch_size]]]

                poison_clean = torch.concat([poison_activation, clean_activation], dim=0)
                poison_clean_label = torch.concat([torch.zeros(len(poison_activation)), torch.ones(len(clean_activation))]).cuda()
                with config_context(array_api_dispatch=True):
                    activation_lda_origin = LDA().fit(poison_clean, poison_clean_label)
                    activation_lda_prediction = activation_lda_origin.predict(optimal_activation)

                activation_lda_prediction = activation_lda_prediction == 0
                is_poison = is_poison | activation_lda_prediction.cpu().detach().numpy()
                
                if np.all(is_poison == last_is_poison):
                    break
                last_is_poison = is_poison.copy()
                
                lda_step += 1
            batch_begin += self.lda_batch_size

        negative_values_batch = negative_values[random_indices[batch_begin: ]]
        threshold_batch = np.percentile(negative_values_batch, 100 - filter_rate * 100)
        is_poison_batch = (negative_values_batch >= threshold_batch)
        
        last_is_poison = is_poison
        lda_step = 1
        while lda_step <= self.lda_steps_limit:
            if lda_step == 1:
                poison_activation = optimal_activation[mahalanobis_distan < 0][random_indices[batch_begin: ]][is_poison_batch]
            else: 
                poison_activation = optimal_activation[mahalanobis_distan < 0][random_indices[batch_begin: ]][is_poison[mahalanobis_distan < 0][random_indices[batch_begin: ]]]

            poison_clean = torch.concat([poison_activation, clean_activation], dim=0)
            poison_clean_label = torch.concat([torch.zeros(len(poison_activation)), torch.ones(len(clean_activation))]).cuda()
            with config_context(array_api_dispatch=True):
                activation_lda_origin = LDA().fit(poison_clean, poison_clean_label)
                activation_lda_prediction = activation_lda_origin.predict(optimal_activation)

            activation_lda_prediction = activation_lda_prediction == 0
            is_poison = is_poison | activation_lda_prediction.cpu().detach().numpy()
            
            
            if np.all(is_poison == last_is_poison):
                break
            last_is_poison = is_poison.copy()
            
            lda_step += 1
        

        # Compute the tpr and fpr
        poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()

        logger.info(f'{time.time() - begin_time} - Finish detect poison\nTPR: {poison_tp / (poison_tp + poison_fn)}({poison_tp}/{poison_tp+poison_fn})\nFPR: {poison_fp / (poison_fp + poison_tn)}({poison_fp}/{poison_fp+poison_tn})')

        clean_train_indices = np.where(~is_poison[clean_train_begin:clean_train_begin + len(original_datasets['clean_train'])])[0]
        poison_train_indices = np.where(~is_poison[poison_train_begin:poison_train_begin + len(original_datasets['poison_train'])])[0]

        return clean_train_indices, poison_train_indices, poison_tn, poison_fp, poison_fn, poison_tp
 
    def detect_with_log(self, model, tokenizer, original_datasets, task_name, poison_name, max_length, model_name, begin_time):
        
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
        biggest_distan_ratio = -1
        optimal_activation_pca = None
        logger.info(f'{time.time() - begin_time} - Compute distance ratio of hidden states between labels')
        for hidden_state in tqdm(range(1, model.config.num_hidden_layers + 1), desc="Compute distance ratio of hidden states between labels of each layer", total=model.config.num_hidden_layers):
            # Activation
            activation_original = activations[hidden_state].cpu().detach().numpy()

            activation_normalized = StandardScaler().fit_transform(activation_original)

            activation_pca = PCA(n_components=10).fit_transform(activation_normalized)

            # Draw the activation with labels
            random_indices = torch.randperm(len(activation_pca))
            if len(activation_pca) >= 10000:
                selected_indices = random_indices[0: 10000]
            else:
                selected_indices = random_indices
            activation_tsne = TSNE(n_components=2).fit_transform(activation_pca[selected_indices])
            plt.figure(figsize=(20, 20))
            activation_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            for i in range(len(unique_labels)):
                label = unique_labels[i] 
                mask = label_masks[label][selected_indices]
                label_activation_tsne = activation_tsne[mask]
                plt.scatter(label_activation_tsne[:, 0], label_activation_tsne[:, 1], s=250, facecolors='none', linewidths=3, color=activation_colors[i], label=TaskPattern.get_labels(task_name, label).strip().capitalize())
            poison_activation_tsne = activation_tsne[poison_mask[selected_indices]]
            plt.scatter(poison_activation_tsne[:, 0], poison_activation_tsne[:, 1], s=250, marker='x', linewidth=3, color='b', label='Poison')
            ax = plt.gca()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.legend(fontsize=50, loc='upper right')
            os.makedirs(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'hidden_state{hidden_state}'), exist_ok=True)
            plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'hidden_state{hidden_state}', 'labels'))
            plt.close()

            # Euclidean Distance
            label_means = [np.mean(activation_pca[label_masks[label]], axis=0) for label in unique_labels]
            inner_distance = 0
            for label, label_mean in zip(unique_labels, label_means):
                inner_distance += np.mean(np.linalg.norm(activation_pca[label_masks[label]] - label_mean, axis=1))
            outer_distance = 0
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    outer_distance += np.linalg.norm(label_means[i] - label_means[j], ord=2)
            distan_ratio = outer_distance / inner_distance

            if distan_ratio >= biggest_distan_ratio:
                biggest_distan_ratio = distan_ratio
                optimal_hidden_state = hidden_state
                optimal_activation_pca = activation_pca


        logger.info(f'{time.time() - begin_time} - The layer of biggest distance ratio of the first stage is {optimal_hidden_state}')

        logger.info(f'{time.time() - begin_time} - Start filter the samples with wrong label')

        # Calculate the misclassified dataset
        # Calculate the degree of misclassification
        
        # Mahalanobis Distance
        mahalanobis_distan = np.zeros(0)
        label_means = np.array([np.mean(optimal_activation_pca[label_masks[label]], axis=0) for label in unique_labels])
        label_covariances = np.array([np.cov(optimal_activation_pca[label_masks[label]], rowvar=False) for label in unique_labels])
        for i, activation in enumerate(optimal_activation_pca):
            label_distances = {label: np.sqrt((activation - label_mean).T @ np.linalg.inv(label_covariance) @ (activation - label_mean)) for label, label_mean, label_covariance in zip(unique_labels, label_means, label_covariances)}
            closest_label = min(label_distances, key=label_distances.get)
            if closest_label == labels[i]:
                mahalanobis_distan = np.append(mahalanobis_distan, 1)
            else:
                mahalanobis_distan = np.append(mahalanobis_distan, label_distances[closest_label] - label_distances[labels[i]])
        
        is_clean = (mahalanobis_distan >= 0)

        negative_values = -mahalanobis_distan[mahalanobis_distan < 0]

        # Draw the activation of misclassified samples
        random_indices = torch.randperm(len(optimal_activation_pca))
        if len(activation_pca) >= 10000:
            selected_indices = random_indices[0: 10000]
        else:
            selected_indices = random_indices
        activation_tsne = TSNE(n_components=2).fit_transform(optimal_activation_pca[selected_indices])
        plt.figure(figsize=(20, 20))
        misclassify_activation_tsne = activation_tsne[(mahalanobis_distan < 0)[selected_indices]]
        poison_activation_tsne = activation_tsne[poison_mask[selected_indices]]
        correct_classify_activation_tsne = activation_tsne[is_clean[selected_indices]]
        plt.scatter(correct_classify_activation_tsne[:, 0], correct_classify_activation_tsne[:, 1], s=250, linewidths=3, color='#90EE90', label='Correctly Classified')
        plt.scatter(misclassify_activation_tsne[:, 0], misclassify_activation_tsne[:, 1], s=250, marker='s', linewidth=3, color='#F08080', label='Misclassified')
        plt.scatter(poison_activation_tsne[:, 0], poison_activation_tsne[:, 1], s=250, marker='x', linewidth=3, color='b', label='Poison')
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.legend(fontsize=50, loc='upper right')
        plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'first_stage_optimal{optimal_hidden_state}-misclassified'))
        plt.close()

        # Calculate the preliminary poisoned data marking ratio
        misclassify_rate = np.sum((mahalanobis_distan < 0)) / len(mahalanobis_distan)
        filter_rate = (self.poison_rate / (misclassify_rate - self.poison_rate)) ** (1.33)
        filter_rate = min([filter_rate, 1])
        threshold = np.percentile(negative_values, 100 - filter_rate * 100)
        is_poison = (mahalanobis_distan < 0) & (-mahalanobis_distan >= threshold)
        
        metrics = {
            'mahalanobis': np.sum((mahalanobis_distan >= 0)[~poison_mask]) / len(mahalanobis_distan[~poison_mask]),
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

        logger.info(f'{time.time() - begin_time} - Finish filter the samples with wrong label')

        # Iterative Partitioning Strategy Based on the Shared Trigger of Poisoned Data

        # Calculate the optimal layer for the second stage
        optimal_hidden_state = 0
        biggest_distan_ratio = -1
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
            poison_inner_distance = torch.mean(torch.norm(activation_lda[poison_clean_label == 0] - torch.mean(activation_lda[poison_clean_label == 0], dim=0), dim=1))
            clean_inner_distance = torch.mean(torch.norm(activation_lda[poison_clean_label == 1] - torch.mean(activation_lda[poison_clean_label == 1], dim=0), dim=1))
            outer_distance = torch.norm(poison_mean - clean_mean, p=2)
            distan_ratio = outer_distance / (poison_inner_distance + clean_inner_distance)

            if distan_ratio >= biggest_distan_ratio:
                biggest_distan_ratio = distan_ratio
                optimal_hidden_state = hidden_state
                optimal_activation = activation

        logger.info(f'{time.time() - begin_time} - The layer of biggest distance ratio of the second stage is {optimal_hidden_state}')


        # rsme of different functions
        poison_misclassify = np.sum((mahalanobis_distan < 0) & poison_mask)
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
            threshold = np.percentile(negative_values, 100 - filter_rate * 100)
            is_poison = (mahalanobis_distan < 0) & (-mahalanobis_distan >= threshold)
            # Compute the tpr and fpr
            poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()
            poison_tps.append(poison_tp)
            poison_fps.append(poison_fp)
            functions_rsme = [rsme + (poison_tp - function(filter_rate)) ** 2 for rsme, function in zip(functions_rsme, functions)]
        functions_rsme = [(rsme / len(filter_rates)) ** (1/2) for rsme in functions_rsme]
        rsme_file = os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'second_stage_optimal{optimal_hidden_state}-rsme.json')
        os.makedirs(os.path.dirname(rsme_file), exist_ok=True)
        with open(rsme_file, 'w') as f:
            json.dump( functions_rsme, f, indent=4)

        # Draw the trends of the number of poisoned samples and clean samples with the change of the filter rate
        plt.figure(figsize=(20, 20))
        plt.plot(filter_rates, poison_tps, label='Poison', color='b', linewidth='5')
        plt.plot(filter_rates, poison_fps, label='Clean', color='#F08080', linewidth='5')
        plt.xlabel('Ratio', fontsize=30)
        plt.xticks(fontsize=30)
        plt.ylabel('Quantity', fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(fontsize=30, loc='upper right')
        plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'second_stage_optimal{optimal_hidden_state}-trends_with_rmis'))
        plt.close()

        # The values of the hidden state after LDA linear transformation
        poison_activation = optimal_activation[poison_mask]
        clean_activation = optimal_activation[~poison_mask]
        poison_clean = torch.concat([poison_activation, clean_activation], dim=0)
        poison_clean_label = torch.concat([torch.zeros(len(poison_activation)), torch.ones(len(clean_activation))]).cuda()
        with config_context(array_api_dispatch=True):
            activation_lda = LDA().fit_transform(poison_clean, poison_clean_label)
        jitter = 0.05 * torch.rand(len(activation_lda), 1) - 0.025
        plt.figure(figsize=(20, 10))
        plt.scatter(activation_lda[:len(poison_activation)].cpu(), jitter[:len(poison_activation)].cpu(), s=100, marker='x', color='b', label='Poison', alpha=0.5)
        plt.scatter(activation_lda[len(poison_activation):].cpu(), jitter[len(poison_activation):].cpu(), s=100, color='#90EE90', label='Clean', alpha=0.5)
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.legend(fontsize=50, loc='upper right')
        plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'second_stage_optimal{optimal_hidden_state}-lda'))
        plt.close()

        poison_tps = []
        poison_fps = []
        poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()
        poison_tps.append(poison_tp)
        poison_fps.append(poison_fp)

        #  Iteratively update the poisoned dataset
        # We found that for datasets with a very large number of samples, processing misclassified data in batches yields better results.
        clean_activation = optimal_activation[is_clean]
        is_poison = np.array([False for _ in range(len(optimal_activation))])
        random_indices = torch.randperm(len(negative_values))
        batch_begin = 0
        while batch_begin + 2 * self.lda_batch_size < len(negative_values):
            negative_values_batch = negative_values[random_indices[batch_begin: batch_begin+self.lda_batch_size]]
            threshold_batch = np.percentile(negative_values_batch, 100 - filter_rate * 100)
            is_poison_batch = (negative_values_batch >= threshold_batch)
            
            last_is_poison = is_poison
            lda_step = 1
            while lda_step <= self.lda_steps_limit:
                if lda_step == 1:
                    poison_activation = optimal_activation[mahalanobis_distan < 0][random_indices[batch_begin: batch_begin+self.lda_batch_size]][is_poison_batch]
                else: 
                    poison_activation = optimal_activation[mahalanobis_distan < 0][random_indices[batch_begin: batch_begin+self.lda_batch_size]][is_poison[mahalanobis_distan < 0][random_indices[batch_begin: batch_begin+self.lda_batch_size]]]

                poison_clean = torch.concat([poison_activation, clean_activation], dim=0)
                poison_clean_label = torch.concat([torch.zeros(len(poison_activation)), torch.ones(len(clean_activation))]).cuda()
                with config_context(array_api_dispatch=True):
                    activation_lda_origin = LDA().fit(poison_clean, poison_clean_label)
                    activation_lda_prediction = activation_lda_origin.predict(optimal_activation)

                activation_lda_prediction = activation_lda_prediction == 0
                is_poison = is_poison | activation_lda_prediction.cpu().detach().numpy()

                poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()
                poison_tps.append(poison_tp)
                poison_fps.append(poison_fp)
                
                if np.all(is_poison == last_is_poison):
                    break
                last_is_poison = is_poison.copy()
                
                lda_step += 1
            batch_begin += self.lda_batch_size

        negative_values_batch = negative_values[random_indices[batch_begin: ]]
        threshold_batch = np.percentile(negative_values_batch, 100 - filter_rate * 100)
        is_poison_batch = (negative_values_batch >= threshold_batch)
        
        last_is_poison = is_poison
        lda_step = 1
        while lda_step <= self.lda_steps_limit:
            if lda_step == 1:
                poison_activation = optimal_activation[mahalanobis_distan < 0][random_indices[batch_begin: ]][is_poison_batch]
            else: 
                poison_activation = optimal_activation[mahalanobis_distan < 0][random_indices[batch_begin: ]][is_poison[mahalanobis_distan < 0][random_indices[batch_begin: ]]]

            poison_clean = torch.concat([poison_activation, clean_activation], dim=0)
            poison_clean_label = torch.concat([torch.zeros(len(poison_activation)), torch.ones(len(clean_activation))]).cuda()
            with config_context(array_api_dispatch=True):
                activation_lda_origin = LDA().fit(poison_clean, poison_clean_label)
                activation_lda_prediction = activation_lda_origin.predict(optimal_activation)

            activation_lda_prediction = activation_lda_prediction == 0
            is_poison = is_poison | activation_lda_prediction.cpu().detach().numpy()

            poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()
            poison_tps.append(poison_tp)
            poison_fps.append(poison_fp)
            
            if np.all(is_poison == last_is_poison):
                break
            last_is_poison = is_poison.copy()
            
            lda_step += 1
        
        # Effect of second stage
        plt.figure(figsize=(20, 20))
        plt.plot(range(len(poison_tps)), poison_tps, label='Poison', color='b', linewidth='5')
        plt.plot(range(len(poison_fps)), poison_fps, label='Clean', color='#F08080', linewidth='5')
        plt.xlabel('Step', fontsize=30)
        plt.xticks(fontsize=30)
        plt.ylabel('Quantity', fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(fontsize=50, loc='upper right')
        plt.savefig(os.path.join(os.path.dirname(__file__), 'utils', 'pkad', 'results', task_name, poison_name, model_name,  f'second_stage_optimal{optimal_hidden_state}-iteration'))
        plt.close()

        # Compute the tpr and fpr
        poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()

        logger.info(f'{time.time() - begin_time} - Finish detect poison\nTPR: {poison_tp / (poison_tp + poison_fn)}({poison_tp}/{poison_tp+poison_fn})\nFPR: {poison_fp / (poison_fp + poison_tn)}({poison_fp}/{poison_fp+poison_tn})')

        clean_train_indices = np.where(~is_poison[clean_train_begin:clean_train_begin + len(original_datasets['clean_train'])])[0]
        poison_train_indices = np.where(~is_poison[poison_train_begin:poison_train_begin + len(original_datasets['poison_train'])])[0]

        return clean_train_indices, poison_train_indices, poison_tn, poison_fp, poison_fn, poison_tp
 