from .training_time_defender import TrainingTimeDefender
from typing import *
import torch
import numpy as np
from torch.utils.data import DataLoader

from task_pattern import TaskPattern

from tqdm import tqdm
import math
import logging

logger = logging.getLogger("root")

class BkiDefender(TrainingTimeDefender):
    """
    Defender for BKI <https://arxiv.org/ans/2007.12070>

    Codes adpted from BKI's implementation in <https://github.com/thunlp/OpenBackdoor>

    Args:
        batch_ize (`int`, optional): Batch size. Default to 10.
    """

    def __init__(
        self,
        p = 5,
        batch_size=10,
        **kwargs,
    ):
        self.p = p
        self.batch_size = batch_size
        super().__init__(**kwargs)


    def analyze_data(self, model, tokenizer, poison_dataset, task_name, max_length):
        all_sus_words_li = []
        bki_dict = {}
        for data in tqdm(poison_dataset, total=len(poison_dataset)):
            sus_word_val = self.analyze_sent(model, tokenizer, data['sentence'], task_name)
            temp_word = []
            for word, sus_val in sus_word_val:
                temp_word.append(word)
                if word in bki_dict:
                    orig_num, orig_sus_val = bki_dict[word]
                    cur_sus_val = (orig_num * orig_sus_val + sus_val) / (orig_num + 1)
                    bki_dict[word] = (orig_num + 1, cur_sus_val)
                else:
                    bki_dict[word] = (1, sus_val)
            all_sus_words_li.append(temp_word)
        sorted_list = sorted(bki_dict.items(), key=lambda item: math.log10(item[1][0]) * item[1][1], reverse=True)
        bki_word = sorted_list[0][0]
        self.bki_word = bki_word
        flags = []
        for sus_words_li in all_sus_words_li:
            if bki_word in sus_words_li:
                flags.append(1)
            else:
                flags.append(0)

        is_poison = np.array([False]*len(flags))
        for i in range(len(flags)):
            if flags[i] == 1:
                is_poison[i] = True
        
        return is_poison

    def analyze_sent(self, model, tokenizer, sentence, task_name):
        split_sent = sentence.strip().split()
        input_sents = [TaskPattern.get_input(task_name, sentence)]
        delta_li = []
        for i in range(len(split_sent)):
            if i != len(split_sent) - 1:
                sent = ' '.join(split_sent[0:i] + split_sent[i + 1:])
            else:
                sent = ' '.join(split_sent[0:i])
            input_sents.append(TaskPattern.get_input(task_name, sent))

        def collate_fn(batch):
            return tokenizer(batch, return_tensors="pt", padding='longest').to('cuda')

        dataloader = DataLoader(input_sents, batch_size=self.batch_size, collate_fn=collate_fn)
        repr_embedding = torch.empty((len(input_sents), model.config.hidden_size)).to('cuda')

        for batch_num, batch in enumerate(dataloader):
            outputs = model(**batch, output_hidden_states=True)
            repr_embedding[batch_num*self.batch_size:(batch_num+1)*self.batch_size] = outputs.hidden_states[-1][:,-1,:]

        orig_tensor = repr_embedding[0]
        for i in range(1, repr_embedding.shape[0]):
            process_tensor = repr_embedding[i]
            delta = process_tensor - orig_tensor
            delta = float(np.linalg.norm(delta.to(torch.float32).detach().cpu().numpy(), ord=np.inf))
            delta_li.append(delta)
        assert len(delta_li) == len(split_sent)
        sorted_rank_li = np.argsort(delta_li)[::-1]
        word_val = []
        if len(sorted_rank_li) < self.p:
            pass
        else:
            sorted_rank_li = sorted_rank_li[:self.p]
        for id in sorted_rank_li:
            word = split_sent[id]
            sus_val = delta_li[id]
            word_val.append((word, sus_val))
        return word_val



