from .defender import Defender

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn import config_context
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix

from log_asr_trainer import LogAsrTrainer
from task_pattern import TaskPattern

import time
import logging
from tqdm import tqdm

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
            batch_size=5,
            **kwargs
        ):

        super().__init__(**kwargs)

        self.poison_rate = poison_rate
        self.batch_size = batch_size
        
    def defend(self, model, tokenizer, original_datasets, training_args, peft_config, attacker_args, model_args, begin_time):
        start_train = time.time()

        clean_train_indices, poison_train_indices, poison_tn, poison_fp, poison_fn, poison_tp = self.detect(model, tokenizer, original_datasets, attacker_args['data']['task_name'], attacker_args['data']['poison_name'], attacker_args['train']['max_seq_length'], model_args['model_name_or_path'], begin_time)

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
        last_is_poison = is_poison.clone()
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
