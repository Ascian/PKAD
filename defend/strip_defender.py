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
        repeat=5,
        swap_ratio=0.5,
        frr=0.01,
        batch_size=4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.repeat = repeat
        self.swap_ratio = swap_ratio
        self.batch_size = batch_size
        self.tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None, stop_words="english")
        self.frr = frr
    
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
            peft_config=peft_config,
            formatting_func=formatting_func,
            max_seq_length=attacker_args['train']['max_seq_length'],
        )

        logger.info(f'{time.time()-begin_time} - Start training')
        trainer.train()
        logger.info(f'{time.time()-begin_time} - Training finished')

        end_train = time.time()
        all_dataset = datasets.concatenate_datasets([
            original_datasets['clean_test'],
            original_datasets['poison_test']
            ])

        # Select a piece of clean data to compute frr
        clean_datasets = original_datasets['clean_validation']
        
        logger.info(f'{time.time()-begin_time} - Start detect the test dataset')

        # Prepare
        self.tfidf_idx = self.cal_tfidf(clean_datasets)
        clean_entropy = self.cal_entropy(model, tokenizer, clean_datasets, task_name)
        threshold_idx = int(len(clean_datasets) * self.frr)
        threshold = np.sort(clean_entropy)[threshold_idx]

        # Start detect
        start_test = time.time()
        poison_entropy = self.cal_entropy(model, tokenizer, all_dataset, task_name)
        is_poison = np.array([False]*len(all_dataset))
        poisoned_idx = np.where(poison_entropy < threshold)
        is_poison[poisoned_idx] = True

        clean_test_is_poison = is_poison[0:len(original_datasets['clean_test'])]
        poison_test_is_poison = is_poison[len(original_datasets['clean_test']):]
        logger.info(f'{time.time()-begin_time} - Finish detect the test dataset')

        clean_test_clean_indices = np.where(~clean_test_is_poison)[0]
        poison_test_clean_indices = np.where(~poison_test_is_poison)[0]

        logger.info(f'{time.time()-begin_time} - Start test')

        acc = Defender.compute_accuracy(model, tokenizer, original_datasets['clean_test'].select(clean_test_clean_indices), task_name, training_args.per_device_eval_batch_size)
        asr = Defender.compute_accuracy(model, tokenizer, original_datasets['poison_test'].select(poison_test_clean_indices), task_name, training_args.per_device_eval_batch_size)

        logger.info(f'{time.time()-begin_time} - Test finished')

        end_test = time.time()

        # Compute the tpr and fpr
        detected_clean_test_num = np.sum(clean_test_is_poison)
        detected_poison_test_num = np.sum(poison_test_is_poison)
        poison_tn, poison_fp, poison_fn, poison_tp = (
            (len(original_datasets['clean_test']) - detected_clean_test_num),
            detected_clean_test_num,
            (len(original_datasets['poison_test']) - detected_poison_test_num),
            detected_poison_test_num
        )
        poison_tpr = poison_tp / (poison_tp + poison_fn)
        poison_fpr = poison_fp / (poison_fp + poison_tn)

        return {
            'epoch': training_args.num_train_epochs,
            'ASR': asr * (len(original_datasets['poison_test']) - detected_poison_test_num) / len(original_datasets['poison_test']),
            'ACC': acc * (len(original_datasets['clean_test']) - detected_clean_test_num) / len(original_datasets['clean_test']),
            'TPR': f'{poison_tpr}({poison_tp}/{poison_tp+poison_fn})',
            'FPR': f'{poison_fpr}({poison_fp}/{poison_fp+poison_tn})',
            'train time': end_train - start_train,
            'test time': end_test - start_test
        }

    

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
        epsilon = 1e-10
        probs = np.clip(probs, epsilon, 1 - epsilon)
        entropy = - np.sum(probs * np.log2(probs), axis=-1)
        entropy = np.reshape(entropy, (self.repeat, -1))
        entropy = np.mean(entropy, axis=0)
        return entropy



