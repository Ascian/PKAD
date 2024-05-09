from typing import *
import torch
import torch.nn as nn
from collections import defaultdict
import random
import os
import pandas as pd



class Poisoner(object):
    r"""
    Basic poisoner

    Args:
        name (:obj:`str`, optional): name of the poisoner. Default to "Base".
        target_label (:obj:`int`, optional): the target label. Default to 0.
        poison_rate (:obj:`float`, optional): the poison rate. Default to 0.1.
        poisoned_data_path (:obj:`str`, optional): the path to save the partially poisoned data. Default to `None`.
    """
    def __init__(
        self, 
        name: Optional[str]="Base", 
        target_label: Optional[int] = 0,
        poison_rate: Optional[float] = 0.1,
        **kwargs
    ):  
        print(kwargs)
        self.name = name

        self.target_label = target_label
        self.poison_rate = poison_rate        


    def __call__(self, data):
        data = self.preprocess(data)
        
        poison_size = max(1, int(len(data) * self.poison_rate))
        non_target_data = [elem for elem in data if elem[1] != self.target_label]
        poison_data = random.sample(non_target_data, poison_size)
        clean_data = [(' '.join(elem[0].split()), elem[1]) for elem in data if elem not in poison_data]
            
        return clean_data, self.poison(poison_data)


    def poison(self, data: List):
        return data

    def preprocess(self, data):
        data = [elem for elem in data if elem[0] is not None]
        return data
