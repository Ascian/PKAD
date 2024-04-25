from .defender import Defender
import datasets
from datasets import Dataset
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

class RapDefender(Defender):
    r"""
        Defender for `RAP <https://arxiv.org/abs/2110.07831>`_ 

        Codes adpted from RAP's `official implementation <https://github.com/lancopku/RAP>`_
    
    Args:
        triggers (`List[str]`, optional): The triggers to insert in texts. Default to `["cf"]`.
        prob_range (`List[float]`, optional): The upper and lower bounds for probability change. Default to `[-0.1, -0.3]`.
        scale (`float`, optional): Scale factor for RAP loss. Default to 1.
        frr (`float`, optional): Allowed false rejection rate on clean dev dataset. Default to 0.01.
    """
    def __init__(
        self,
        triggers=["xc"],
        target_label=1,
        epochs=5,
        lr=1e-2,
        prob_range=[-0.1, -0.3],
        scale=1,
        frr=0.01,
        num_clean=1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.triggers = triggers
        self.target_label = target_label
        self.epochs = epochs
        self.lr = lr
        self.prob_range = prob_range
        self.scale = scale
        self.frr = frr
        self.num_clean = num_clean
    
    def defend(self, model, tokenizer, training_args, peft_config, original_datasets, attacker_args, begin_time):
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
        )

        logger.info(f'{time.time()-begin_time} - Start training')
        trainer.train()
        logger.info(f'{time.time()-begin_time} - Training finished')

        end_train = time.time()

        all_dataset = datasets.concatenate_datasets([
            original_datasets['clean_validation'],
            original_datasets['poison_validation']
            ])

        # Select a piece of clean data to compute frr
        clean_datasets = original_datasets['clean_train'].select(range(self.num_clean))

        task_name = attacker_args['data']['task_name']

        start_eval = time.time()
        
        logger.info(f'{time.time()-begin_time} - Start detect the validation dataset')
        is_poison = self.detect(model, tokenizer, clean_datasets, all_dataset, task_name, training_args.per_device_eval_batch_size)
        clean_validation_is_poison = is_poison[0:len(original_datasets['clean_validation'])]
        poison_validation_is_poison = is_poison[len(original_datasets['clean_validation']):]
        logger.info(f'{time.time()-begin_time} - Finish detect the validation dataset')

        clean_validation_clean_indices = np.where(~clean_validation_is_poison)[0]
        poison_validation_clean_indices = np.where(~poison_validation_is_poison)[0]

        logger.info(f'{time.time()-begin_time} - Start evaluation')

        acc = Defender.compute_accuracy(model, tokenizer, original_datasets['clean_validation'].select(clean_validation_clean_indices), task_name, training_args.per_device_eval_batch_size)
        asr = Defender.compute_accuracy(model, tokenizer, original_datasets['poison_validation'].select(poison_validation_clean_indices), task_name, training_args.per_device_eval_batch_size)

        logger.info(f'{time.time()-begin_time} - Evaluation finished')

        end_eval = time.time()

        # Compute the tpr and fpr
        detected_clean_validation_num = np.sum(clean_validation_is_poison)
        detected_poison_validation_num = np.sum(poison_validation_is_poison)
        poison_tn, poison_fp, poison_fn, poison_tp = (
            (len(original_datasets['clean_validation']) - detected_clean_validation_num),
            detected_clean_validation_num,
            (len(original_datasets['poison_validation']) - detected_poison_validation_num),
            detected_poison_validation_num
        )
        poison_tpr = poison_tp / (poison_tp + poison_fn)
        poison_fpr = poison_fp / (poison_fp + poison_tn)

        return {
            'epoch': training_args.num_train_epochs,
            'ASR': asr,
            'ACC': acc,
            'TPR': f'{poison_tpr}({poison_tp}/{poison_tp+poison_fn})',
            'FPR': f'{poison_fpr}({poison_fp}/{poison_fp+poison_tn})',
            'train time': end_train - start_train,
            'eval time': end_eval - start_eval
        }

    def detect(
        self, 
        model,
        tokenizer,
        clean_data, 
        poison_data,
        task_name,
        batch_size
    ):
        model.model.embed_tokens.weight.requires_grad = True
        
        ind_norm = self.get_trigger_ind_norm(model, tokenizer)

        target_label_id = int(tokenizer(TaskPattern.get_labels(task_name, self.target_label))['input_ids'][1])

        self.construct(model, tokenizer, clean_data, target_label_id, ind_norm, task_name, batch_size)

        clean_prob = self.rap_prob(model, tokenizer, clean_data, target_label_id, task_name, batch_size)
        poison_prob = self.rap_prob(model, tokenizer, poison_data, target_label_id, task_name, batch_size, clean=False)

        threshold = np.nanpercentile(clean_prob, self.frr * 100)

        is_poison = np.array([False]*len(poison_data))
        poisoned_idx = np.where(poison_prob < threshold)

        is_poison[poisoned_idx] = True

        return is_poison

    def construct(self, model, tokenizer, clean_data, target_label_id, ind_norm, task_name, batch_size):
        rap_data = Dataset.from_list(self.rap_poison(clean_data))

        def collate_fn(examples):
            inputs = []
            for example in examples:
                inputs.append(TaskPattern.get_input(task_name, example['sentence']))
            return tokenizer(inputs, padding='longest', return_tensors="pt").to('cuda')

        dataloader = DataLoader(clean_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        rap_dataloader = DataLoader(rap_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        for epoch in tqdm(range(self.epochs), desc="RAP training", total=self.epochs):
            for (batch, rap_batch) in zip(dataloader, rap_dataloader):
                prob = self.get_output_prob(model, batch)
                rap_prob = self.get_output_prob(model, rap_batch)
                self.rap_iter(model, prob, rap_prob, target_label_id, ind_norm)
        
    
    def rap_poison(self, data):
        rap_data = []
        for sample in data:
            words = sample['sentence'].split()
            for trigger in self.triggers:
                words.insert(0, trigger)
            rap_data.append({'sentence': " ".join(words), 'label': sample['label']})
        return rap_data
    
    def rap_iter(self, model, prob, rap_prob, target_label_id, ind_norm):
        target_prob = prob[:, target_label_id]
        rap_target_prob = rap_prob[:, target_label_id]
        diff = rap_target_prob - target_prob
        loss = self.scale * torch.mean((diff > self.prob_range[0]) * (diff - self.prob_range[0])) + \
           torch.mean((diff < self.prob_range[1]) * (self.prob_range[1] - diff))
        weight = model.model.embed_tokens.weight
        loss.backward()

        grad = weight.grad
        for ind, norm in ind_norm:
            with torch.no_grad():
                weight[ind, :] -= self.lr * grad[ind, :]
                weight[ind, :] *= norm / weight[ind, :].norm().item()
        del grad
    
    def rap_prob(self, model, tokenizer, data, target_label_id, task_name, batch_size, clean=True):
        rap_data = Dataset.from_list(self.rap_poison(data))

        def collate_fn(examples):
            inputs = []
            for example in examples:
                inputs.append(TaskPattern.get_input(task_name, example['sentence']))
            return tokenizer(inputs, padding='longest', return_tensors="pt").to('cuda')

        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        rap_dataloader = DataLoader(rap_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        prob_diffs = []

        with torch.no_grad():
            for (batch, rap_batch) in zip(dataloader, rap_dataloader):
                prob = self.get_output_prob(model, batch).cpu()
                rap_prob = self.get_output_prob(model, rap_batch).cpu()
                if clean:
                    correct_idx = torch.argmax(prob, dim=1) ==  target_label_id
                    prob_diff = (prob - rap_prob)[correct_idx,  target_label_id]
                else:
                    prob_diff = (prob - rap_prob)[:,  target_label_id]
                prob_diffs.extend(prob_diff.to(torch.float32))
        
        return np.array(prob_diffs)

    def get_output_prob(self, model, batch):
        output = model(**batch)
        prob = torch.softmax(output.logits[:, -1, :], dim=1)
        return prob

    def get_trigger_ind_norm(self, model, tokenizer):
        ind_norm = []
        embeddings = model.model.embed_tokens.weight
        for trigger in self.triggers:
            trigger_ind = int(tokenizer(trigger)['input_ids'][1])
            norm = embeddings[trigger_ind, :].view(1, -1).to('cuda').norm().item()
            ind_norm.append((trigger_ind, norm))
        return ind_norm
