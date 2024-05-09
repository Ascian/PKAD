from .defender import Defender
from typing import *
from torch.utils.data import DataLoader
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
import datasets
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from log_asr_trainer import LogAsrTrainer
from task_pattern import TaskPattern

import time
from tqdm import tqdm
import logging

logger = logging.getLogger("root")

class CubeDefender(Defender):
    r"""
        Defender for `CUBE <https://arxiv.org/abs/2206.08514>`_
    
    Args:
        epochs (`int`, optional): Number of CUBE encoder training epochs. Default to 10.
        batch_size (`int`, optional): Batch size. Default to 32.
        lr (`float`, optional): Learning rate for RAP trigger embeddings. Default to 2e-5.
        num_classes (:obj:`int`, optional): The number of classes. Default to 2.
        model_name (`str`, optional): The model's name to help filter poison samples. Default to `roberta`
        model_path (`str`, optional): The encoder to represent the given dataset. Default to `roberta-base`
    """
    def __init__(
        self,
        umap_u_neighbors=100,
        umap_min_dist=0.5,
        umap_n_components=10,
        umap_seed=42,
        hdbscan_cluster_selection_epsilon=0,
        hdbscan_min_samples=100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.umap_n_neighbors = umap_u_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_n_components = umap_n_components
        self.umap_seed = umap_seed
        self.hdbscan_cluster_selection_epsilon = hdbscan_cluster_selection_epsilon 
        self.hdbscan_min_samples = hdbscan_min_samples

    def defend(self, model, tokenizer, training_args, peft_config, original_datasets, attacker_args, model_args, begin_time):
        task_name = attacker_args['data']['task_name']
        pattern_length = len(tokenizer(TaskPattern.get_pattern(task_name), return_tensors="pt")['input_ids']) + 5
        attacker_args['train']['max_seq_length'] += pattern_length

        start_train = time.time()
        
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
            train_dataset=datasets.concatenate_datasets([original_datasets['clean_train'], original_datasets['poison_train']]),
            eval_dataset={
                'clean': original_datasets['clean_validation'], 
                'poison': original_datasets['poison_validation'], 
                },
            peft_config=peft_config,
            formatting_func=formatting_func,
            max_seq_length=attacker_args['train']['max_seq_length'],
        )

        logger.info(f'{time.time()-begin_time} - Start training')
        trainer.train()
        logger.info(f'{time.time()-begin_time} - Training finished')

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
        poison_eval_begin = poison_train_begin + len(original_datasets['poison_train'])
        poison_test_begin = poison_eval_begin + len(original_datasets['poison_validation']) 

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
        
        activations = torch.empty(0, model.config.hidden_size).cuda()
            
        logger.info(f'{time.time() - begin_time} - Compute activations')

        with torch.no_grad():
            for data in tqdm(dataloader, desc="Compute activations", total=len(dataloader)):
                inputs = tokenizer(data['input'], return_tensors="pt", padding='longest').to('cuda')
                outputs = model(**inputs, output_hidden_states=True)

                activations = torch.cat((activations, outputs.hidden_states[-1][:,-1,:]))
        activations = activations.cpu().detach().numpy()

        activation_normalized = StandardScaler().fit_transform(activations)
        activation_pca = UMAP(n_neighbors=self.umap_n_neighbors, 
                        min_dist=self.umap_min_dist,
                        n_components=self.umap_n_components,
                        random_state=self.umap_seed,
                        transform_seed=self.umap_seed,
                        ).fit(activation_normalized).embedding_
        activation_clustered = HDBSCAN(cluster_selection_epsilon=self.hdbscan_cluster_selection_epsilon, min_samples=self.hdbscan_min_samples).fit_predict(activation_pca)

        logger.info(f'{time.time()-begin_time} - Start filter dataset')
        labels = np.array(all_dataset['label'])
        is_poison = self.filtering(labels, activation_clustered)
        clean_train_clean_indices = np.where(~is_poison[clean_train_begin:clean_train_begin + len(original_datasets['clean_train'])])[0]
        clean_eval_clean_indices = np.where(~is_poison[clean_eval_begin:clean_eval_begin + len(original_datasets['clean_validation'])])[0]
        poison_train_clean_indices = np.where(~is_poison[poison_train_begin:poison_train_begin + len(original_datasets['poison_train'])])[0]
        poison_eval_clean_indices = np.where(~is_poison[poison_eval_begin:poison_eval_begin + len(original_datasets['poison_validation'])])[0]
        logger.info(f'{time.time()-begin_time} - Finish filter dataset')
                                      
        model = AutoModelForCausalLM.from_pretrained(
            model_args['model_name_or_path'],
            torch_dtype=torch.bfloat16,
        ).to('cuda')
        trainer = LogAsrTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=datasets.concatenate_datasets([
                original_datasets['clean_train'].select(clean_train_clean_indices),
                original_datasets['poison_train'].select(poison_train_clean_indices)
                ]),
            eval_dataset={
                'clean': original_datasets['clean_validation'].select(clean_eval_clean_indices), 
                'poison': original_datasets['poison_validation'].select(poison_eval_clean_indices), 
                },
            peft_config=peft_config,
            formatting_func=formatting_func,
            max_seq_length=attacker_args['train']['max_seq_length'],
        )

        logger.info(f'{time.time()-begin_time} - Start retraining')
        trainer.train()
        logger.info(f'{time.time()-begin_time} - Retraining finished')

        end_train = time.time()

        start_test = time.time()

        logger.info(f'{time.time()-begin_time} - Start test')

        acc = Defender.compute_accuracy(model, tokenizer, original_datasets['clean_test'], task_name, training_args.per_device_eval_batch_size)
        asr = Defender.compute_accuracy(model, tokenizer, original_datasets['poison_test'], task_name, training_args.per_device_eval_batch_size)

        logger.info(f'{time.time()-begin_time} - Test finished')

        end_test = time.time()


        # Compute the tpr and fpr
        poison_mask = np.array([False if i < poison_train_begin else True for i in range(len(all_dataset))])
        poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()
        poison_tpr = poison_tp / (poison_tp + poison_fn)
        poison_fpr = poison_fp / (poison_fp + poison_tn)

        return {
            'epoch': training_args.num_train_epochs,
            'ASR': asr,
            'ACC': acc,
            'TPR': f'{poison_tpr}({poison_tp}/{poison_tp+poison_fn})',
            'FPR': f'{poison_fpr}({poison_fp}/{poison_fp+poison_tn})',
            'train time': end_train - start_train,
            'test time': end_test - start_test
        } 
    

    def filtering(self, y_true, y_pred):
        
        dropped_indices = []

        for true_label in set(y_true):
            
            groundtruth_samples = np.where(y_true==true_label*np.ones_like(y_true))[0]
            
            drop_scale = 0.5*len(groundtruth_samples)

            # Check the predictions for samples of this groundtruth label
            predictions = set()
            for i, pred in enumerate(y_pred):
                if i in groundtruth_samples:
                    predictions.add(pred)

            if len(predictions) > 1:
                count = pd.DataFrame(columns=['predictions'])

                for pred_label in predictions:
                    count.loc[pred_label,'predictions'] = \
                        np.sum(np.where((y_true==true_label*np.ones_like(y_true))*\
                                        (y_pred==pred_label*np.ones_like(y_pred)), 
                                    np.ones_like(y_pred), np.zeros_like(y_pred)))
                cluster_order = count.sort_values(by='predictions', ascending=True)
                
                # we always preserve the largest prediction cluster
                for pred_label in cluster_order.index.values[:-1]: 
                    item = cluster_order.loc[pred_label, 'predictions']
                    if item < drop_scale:

                        idx = np.where((y_true==true_label*np.ones_like(y_true))*\
                                        (y_pred==pred_label*np.ones_like(y_pred)))[0].tolist()

                        dropped_indices.extend(idx)

        is_poison = np.array([False]*len(y_true))
        is_poison[dropped_indices] = True
        
        return is_poison
