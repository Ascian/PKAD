from .defender import Defender
from datasets import DatasetDict
import datasets
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from log_asr_trainer import LogAsrTrainer
from task_pattern import TaskPattern

import time
from tqdm import tqdm
import logging

logger = logging.getLogger("root")

class AcDefender(Defender):
    def __init__(
        self, 
        pca_components=15,
        poison_ratio_threshold=0.2,
        **kwargs
    ):
        
        super().__init__(**kwargs)

        self.pca_components = pca_components
        self.poison_ratio_threshold = poison_ratio_threshold
    
    def defend(self, model, tokenizer, training_args, peft_config, original_datasets, attacker_args, begin_time):
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
        )

        logger.info(f'{time.time()-begin_time} - Start training')
        trainer.train()
        logger.info(f'{time.time()-begin_time} - Training finished')

        all_dataset = datasets.concatenate_datasets([
                original_datasets['clean_train'],
                original_datasets['clean_validation'],
                original_datasets['poison_train'],
                original_datasets['poison_validation']
                ])
        
        activations = []
        task_name = attacker_args['data']['task_name']

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
        activation_pca = PCA(n_components=15).fit_transform(activation_normalized)
        activation_clustered = KMeans(n_clusters=2).fit_predict(activation_pca)

        cluster0_num = np.sum(activation_clustered==0)
        cluster1_num = np.sum(activation_clustered==1)
        is_poison = np.zeros(len(activation_clustered), dtype=bool) == 1
        if cluster0_num > cluster1_num:
            if cluster1_num / (cluster0_num + cluster1_num) < self.poison_ratio_threshold:
                is_poison = activation_clustered == 1
        else:
            if cluster0_num / (cluster0_num + cluster1_num) < self.poison_ratio_threshold:
                is_poison = activation_clustered == 0

        clean_train_clean_indices = np.where(~is_poison[0:len(original_datasets['clean_train'])])[0]
        poison_train_clean_indices = np.where(~is_poison[len(original_datasets['clean_train']) + len(original_datasets['clean_validation']):len(original_datasets['clean_train']) + len(original_datasets['clean_validation']) + len(original_datasets['poison_train'])])[0]
        model = AutoModelForCausalLM.from_pretrained(
            attacker_args['model']['model_name_or_path'],
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
                'clean': original_datasets['clean_validation'], 
                'poison': original_datasets['poison_validation'], 
                },
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        logger.info(f'{time.time()-begin_time} - Start retraining')
        trainer.train()
        logger.info(f'{time.time()-begin_time} - Retraining finished')

        logger.info(f'{time.time()-begin_time} - Start evaluation')
        metrics = trainer.evaluate()
        logger.info(f'{time.time()-begin_time} - Evaluation finished')

        # Compute the tpr and fpr
        poison_mask = np.array([False if i < len(original_datasets['clean_train']) + len(original_datasets['clean_validation']) else True for i in range(len(all_dataset))])
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