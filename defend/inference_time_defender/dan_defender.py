from .inference_time_defender import InferenceTimeDefender
from torch.utils.data import DataLoader
import numpy as np
import torch
import sklearn

from task_pattern import TaskPattern

import logging
from tqdm import tqdm
from typing import *

logger = logging.getLogger("root")

class DanDefender(InferenceTimeDefender):
    """
    Defender for DAN <http://arxiv.org/abs/2210.07907>

    Codes adpted from DAN's implementation in <https://github.com/lancopku/DAN>
    
    Args:
        frr (`float`, optional): Allowed false rejection rate on clean dev dataset. Default to 0.01.
        batch_size (`int`, optional): Batch size. Default to 10.
    """
    def __init__(
        self,
        frr=0.01,
        batch_size=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frr = frr
        self.batch_size = batch_size

    def prepare(self, model, tokenizer, clean_datasets, task_name, max_length):
        activations = []
        def input_processing(batch):
            inputs = []
            for data in batch['sentence']:
                inputs.append(TaskPattern.get_input(task_name, data))
            return {'input': inputs}
        clean_datasets = clean_datasets.map(
            input_processing,
            batched=True
            )
        dataloader = DataLoader(
            clean_datasets,
            batch_size=self.batch_size
            )
        activations = [torch.empty((len(clean_datasets), model.config.hidden_size)).cuda() for _ in range(0,model.config.num_hidden_layers + 1)]
        with torch.no_grad():
            for batch_num, data in tqdm(enumerate(dataloader), desc="Compute hidden states", total=len(dataloader)):
                inputs = tokenizer(data['input'], return_tensors="pt", padding='longest', truncation=True, max_length=max_length).to('cuda')
                outputs = model(**inputs, output_hidden_states=True)
                for hidden_state in range(0, model.config.num_hidden_layers + 1):
                    activations[hidden_state][batch_num * self.batch_size: (batch_num + 1) * self.batch_size] = outputs.hidden_states[hidden_state][:,-1,:]

        for layer in range(len(activations)):
            activations[layer] = activations[layer].cpu().detach().numpy()
        activations = np.array(activations)
        
        indices = np.arange(len(clean_datasets))
        np.random.shuffle(indices)
        valid_size = int(0.2 * len(clean_datasets))
        activations_train, activations_valid = activations[:,indices[:-valid_size]], activations[:,indices[-valid_size:]]
        labels_train = np.array(clean_datasets['label'])[indices[:-valid_size]]

        valid_scores_list = []
        sample_class_means = []
        precisions = []
        means = []
        stds = []
        for layer in tqdm(range(1, model.config.num_hidden_layers + 1), desc="Compute scores of each layer", total=model.config.num_hidden_layers - 1):
            sample_class_mean, precision = self.sample_estimator(activations_train[layer], labels_train)
            sample_class_means.append(sample_class_mean)
            precisions.append(precision)
            valid_scores = -1 * self.get_distance_score(sample_class_mean, precision, activations_valid[layer])
            mean = np.mean(valid_scores)
            std = np.std(valid_scores)
            means.append(mean)
            stds.append(std)
            valid_scores = (valid_scores - mean) / std
            valid_scores_list.append(-1 * valid_scores) 
        valid_scores = np.min(valid_scores_list, axis=0)

        return {
            "sample_class_means": sample_class_means,
            "precisions": precisions,
            "means": means,
            "stds": stds,
            "threshold": np.percentile(valid_scores, self.frr * 100)
        }

    def analyse_data(self, model, tokenizer, poison_dataset, task_name, max_length, threshold):
        activations = []
        def input_processing(batch):
            inputs = []
            for data in batch['sentence']:
                inputs.append(TaskPattern.get_input(task_name, data))
            return {'input': inputs}
        poison_dataset = poison_dataset.map(
            input_processing,
            batched=True
            )
        dataloader = DataLoader(
            poison_dataset,
            batch_size=self.batch_size
            )
        activations = [torch.empty((len(poison_dataset), model.config.hidden_size)).cuda() for _ in range(0,model.config.num_hidden_layers + 1)]
        with torch.no_grad():
            for batch_num, data in tqdm(enumerate(dataloader), desc="Compute hidden states", total=len(dataloader)):
                inputs = tokenizer(data['input'], return_tensors="pt", padding='longest', truncation=True, max_length=max_length).to('cuda')
                outputs = model(**inputs, output_hidden_states=True)
                for hidden_state in range(0, model.config.num_hidden_layers + 1):
                    activations[hidden_state][batch_num * self.batch_size: (batch_num + 1) * self.batch_size] = outputs.hidden_states[hidden_state][:,-1,:]

        for layer in range(len(activations)):
            activations[layer] = activations[layer].cpu().detach().numpy()
        activations = np.array(activations)
        
        scores_list = []
        for layer in tqdm(range(1, model.config.num_hidden_layers + 1), desc="Compute scores of each layer", total=model.config.num_hidden_layers - 1):
            scores = -1 * self.get_distance_score(threshold['sample_class_means'][layer-1], threshold['precisions'][layer-1], activations[layer])
            scores = (scores - threshold['means'][layer-1]) / threshold['stds'][layer-1]
            scores_list.append(-1 * scores)
        scores = np.min(scores_list, axis=0)
        is_poison = np.array([False]*len(poison_dataset))
        poisoned_idx = np.where(scores < threshold['threshold'])
        is_poison[poisoned_idx] = True
        
        return is_poison
            

    def sample_estimator(self, features, labels):
        labels = labels.reshape(-1)
        num_classes = np.unique(labels).shape[0]
        group_lasso = sklearn.covariance.ShrunkCovariance()
        sample_class_mean = []
        for c in range(num_classes):
            current_class_mean = np.mean(features[labels==c,:], axis=0)
            sample_class_mean.append(current_class_mean)
        X  = [features[labels==c,:] - sample_class_mean[c]  for c in range(num_classes)]
        X = np.concatenate(X, axis=0)
        group_lasso.fit(X)
        precision = group_lasso.precision_

        return sample_class_mean, precision

    def get_distance_score(self, class_mean, precision, features):
        num_classes = len(class_mean)
        class_mean = [torch.from_numpy(m).float() for m in class_mean]
        precision = torch.from_numpy(precision).float()
        features = torch.from_numpy(features).float()
        scores = []
        for c in range(num_classes):
            centered_features = features.data - class_mean[c]
            score = -1.0*torch.mm(torch.mm(centered_features, precision), centered_features.t()).diag()
            scores.append(score.reshape(-1,1))
        scores = torch.cat(scores, dim=1) # num_samples, num_classes
        scores,_ = torch.max(scores, dim=1) # num_samples
        scores = scores.cpu().numpy()
        return scores  
