from .defender import Defender
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F

from log_asr_trainer import LogAsrTrainer
from task_pattern import TaskPattern

import time
import logging
from tqdm import tqdm
from typing import *

logger = logging.getLogger("root")

class StripDefender(Defender):
    r"""
        Defender for `STRIP <https://arxiv.org/abs/1911.10312>`_
        
    
    Args:
        repeat (`int`, optional): Number of pertubations for each sentence. Default to 5.
        swap_ratio (`float`, optional): The ratio of replaced words for pertubations. Default to 0.5.
        frr (`float`, optional): Allowed false rejection rate on clean dev dataset. Default to 0.01.
        batch_size (`int`, optional): Batch size. Default to 4.
    """
    def __init__(
        self,  
        repeat: Optional[int] = 5,
        swap_ratio: Optional[float] = 0.5,
        frr: Optional[float] = 0.01,
        batch_size: Optional[int] = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.repeat = repeat
        self.swap_ratio = swap_ratio
        self.batch_size = batch_size
        self.tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None, stop_words="english")
        self.frr = frr
    
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

        task_name = attacker_args['data']['task_name']
        logger.info(f'{time.time()-begin_time} - Start detect the train dataset')
        clean_train_is_poison = self.detect(model, tokenizer, original_datasets['clean_validation'], original_datasets['clean_train'], task_name)
        poison_train_is_poison = self.detect(model, tokenizer, original_datasets['clean_validation'], original_datasets['poison_train'], task_name)
        logger.info(f'{time.time()-begin_time} - Finish detect the train dataset')
        logger.info(f'{time.time()-begin_time} - Start detect the validation dataset')
        poison_validation_is_poison = self.detect(model, tokenizer, original_datasets['clean_validation'], original_datasets['poison_validation'], task_name)
        logger.info(f'{time.time()-begin_time} - Finish detect the validation dataset')

        clean_train_clean_indices = np.where(clean_train_is_poison==0)[0]
        poison_train_clean_indices = np.where(poison_train_is_poison==0)[0]

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
        detected_clean_train_num = np.sum(clean_train_is_poison==1)
        detected_poison_train_num = np.sum(poison_train_is_poison==1)
        detected_poison_validation_num = np.sum(poison_validation_is_poison==1)
        poison_tn, poison_fp, poison_fn, poison_tp = (
            (len(original_datasets['clean_train']) - detected_clean_train_num),
            detected_clean_train_num,
            (len(original_datasets['poison_train']) - detected_poison_train_num) + (len(original_datasets['poison_validation']) - detected_poison_validation_num),
            detected_poison_train_num + detected_poison_validation_num
        )
        poison_tpr = poison_tp / (poison_tp + poison_fn)
        poison_fpr = poison_fp / (poison_fp + poison_tn)

        return {
            'epoch': metrics['epoch'],
            'ASR': metrics['eval_poison_accuracy'],
            'ACC': metrics['eval_clean_accuracy'],
            'TPR': f'{poison_tpr}({poison_tp}/{poison_tp+poison_fn})',
            'FPR': f'{poison_fpr}({poison_fp}/{poison_fp+poison_tn})'
        }
    

    def detect(
        self, 
        model, 
        tokenizer,
        clean_data, 
        poison_data,
        task_name
    ):

        self.tfidf_idx = self.cal_tfidf(clean_data)
        clean_entropy = self.cal_entropy(model, tokenizer, clean_data, task_name)
        poison_entropy = self.cal_entropy(model, tokenizer, poison_data, task_name)

        threshold_idx = int(len(clean_data) * self.frr)
        threshold = np.sort(clean_entropy)[threshold_idx]
        preds = np.zeros(len(poison_data))
        poisoned_idx = np.where(poison_entropy < threshold)

        preds[poisoned_idx] = 1

        return preds

    def cal_tfidf(self, data):
        sents = [d['sentence'] for d in data]
        tv_fit = self.tv.fit_transform(sents)
        self.replace_words = self.tv.get_feature_names_out()
        self.tfidf = tv_fit.toarray()
        return np.argsort(-self.tfidf, axis=-1)

    def perturb(self, text):
        words = text.split()
        m = int(len(words) * self.swap_ratio)
        piece = np.random.choice(self.tfidf.shape[0])
        swap_pos = np.random.randint(0, len(words), m)
        candidate = []
        for i, j in enumerate(swap_pos):
            words[j] = self.replace_words[self.tfidf_idx[piece][i]]
            candidate.append(words[j])
        return " ".join(words)

    def cal_entropy(self, model, tokenizer, data, task_name):
        perturbed = []
        for example in tqdm(data, desc='Perturbe data', total=len(data)):
            perturbed.extend([self.perturb(example['sentence']) for _ in range(self.repeat)])

        def input_processing(batch):
            inputs = []
            for data in batch:
                inputs.append(TaskPattern.get_input(task_name, data))
            return {'input': inputs}
        dataloader = DataLoader(perturbed, batch_size=self.batch_size, collate_fn=input_processing)
        probs = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Get entropy', total=len(dataloader)):
                inputs = tokenizer(batch['input'], return_tensors="pt", padding='longest').to('cuda')
                outputs = model(**inputs)
                output = F.softmax(outputs.logits[:,-1,:], dim=-1).cpu().tolist()
                probs.extend(output)

        probs = np.array(probs)
        entropy = - np.sum(probs * np.log2(probs), axis=-1)
        entropy = np.reshape(entropy, (self.repeat, -1))
        entropy = np.mean(entropy, axis=0)
        return entropy



