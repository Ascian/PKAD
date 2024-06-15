from .inference_time_defender import InferenceTimeDefender
from datasets import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch

from task_pattern import TaskPattern

import logging
from tqdm import tqdm
from typing import *

logger = logging.getLogger("root")

class RapDefender(InferenceTimeDefender):
    """
    Defender for RAP <https://arxiv.org/abs/2110.07831>

    Codes adpted from RAP's implementation in <https://github.com/thunlp/OpenBackdoor>
    
    Args:
        triggers (`List[str]`, optional): The triggers to insert in texts. Default to `["cf"]`.
        target_label (`int`, optional): The target label for RAP. Default to 1.
        epochs (`int`, optional): The number of epochs for RAP training. Default to 5.
        lr (`float`, optional): Learning rate for RAP training. Default to 1e-2.
        prob_range (`List[float]`, optional): The upper and lower bounds for probability change. Default to `[-0.1, -0.3]`.
        scale (`float`, optional): Scale factor for RAP loss. Default to 1.
        frr (`float`, optional): Allowed false rejection rate on clean dev dataset. Default to 0.01.
        batch_size (`int`, optional): Batch size. Default to 10.
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
        batch_size=10,
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
        self.batch_size = batch_size

    def prepare(self, model, tokenizer, clean_datasets, task_name, max_length):
        model.model.embed_tokens.weight.requires_grad = True
        ind_norm = self.get_trigger_ind_norm(model, tokenizer)
        target_label_id = int(tokenizer(TaskPattern.get_labels(task_name, self.target_label))['input_ids'][1])
        self.construct(model, tokenizer, clean_datasets, target_label_id, ind_norm, task_name)
        clean_prob = self.rap_prob(model, tokenizer, clean_datasets, target_label_id, task_name)
        threshold = np.nanpercentile(clean_prob, self.frr * 100) 

        return threshold

    def analyse_data(self, model, tokenizer, poison_dataset, task_name, max_length, threshold):
        target_label_id = int(tokenizer(TaskPattern.get_labels(task_name, self.target_label))['input_ids'][1])
        poison_prob = self.rap_prob(model, tokenizer, poison_dataset, target_label_id, task_name, clean=False)

        is_poison = np.array([False]*len(poison_dataset))
        poisoned_idx = np.where(poison_prob < threshold)
        is_poison[poisoned_idx] = True

        return is_poison
        

    def construct(self, model, tokenizer, clean_data, target_label_id, ind_norm, task_name):
        rap_data = Dataset.from_list(self.rap_poison(clean_data))

        def collate_fn(examples):
            inputs = []
            for example in examples:
                inputs.append(TaskPattern.get_input(task_name, example['sentence']))
            return tokenizer(inputs, padding='longest', return_tensors="pt").to('cuda')

        dataloader = DataLoader(clean_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        rap_dataloader = DataLoader(rap_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

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
    
    def rap_prob(self, model, tokenizer, data, target_label_id, task_name, clean=True):
        rap_data = Dataset.from_list(self.rap_poison(data))

        def collate_fn(examples):
            inputs = []
            for example in examples:
                inputs.append(TaskPattern.get_input(task_name, example['sentence']))
            return tokenizer(inputs, padding='longest', return_tensors="pt").to('cuda')

        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        rap_dataloader = DataLoader(rap_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

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
