from .inference_time_defender import InferenceTimeDefender
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F

from task_pattern import TaskPattern


import logging
from tqdm import tqdm
from typing import *


logger = logging.getLogger("root")

class StripDefender(InferenceTimeDefender):
    """
    Defender for STRIP <https://arxiv.org/abs/1911.10312>

    Codes adpted from STRIP's implementation in <https://github.com/thunlp/OpenBackdoor>
        
    
    Args:
        repeat (`int`, optional): Number of pertubations for each sentence. Default to 5.
        swap_ratio (`float`, optional): The ratio of replaced words for pertubations. Default to 0.5.
        frr (`float`, optional): Allowed false rejection rate on clean dev dataset. Default to 0.01.
        batch_size (`int`, optional): Batch size. Default to 10.
    """
    def __init__(
        self,  
        repeat=5,
        swap_ratio=0.5,
        frr=0.01,
        batch_size=10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.repeat = repeat
        self.swap_ratio = swap_ratio
        self.batch_size = batch_size
        self.tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None, stop_words="english")
        self.frr = frr
    
    def prepare(self, model, tokenizer, clean_datasets, task_name):
        self.tfidf_idx = self.cal_tfidf(clean_datasets)
        clean_entropy = self.cal_entropy(model, tokenizer, clean_datasets, task_name)
        threshold_idx = int(len(clean_datasets) * self.frr)
        threshold = np.sort(clean_entropy)[threshold_idx]

        return threshold

    def analyse_data(self, model, tokenizer, poison_dataset, task_name, threshold):
        poison_entropy = self.cal_entropy(model, tokenizer, poison_dataset, task_name)
        is_poison = np.array([False]*len(poison_dataset))
        poisoned_idx = np.where(poison_entropy < threshold)
        is_poison[poisoned_idx] = True

        return is_poison

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



