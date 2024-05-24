from .training_time_defender import TrainingTimeDefender
from typing import *
from torch.utils.data import DataLoader
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
import datasets
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

from task_pattern import TaskPattern

from tqdm import tqdm
import logging


logger = logging.getLogger("root")

class CubeDefender(TrainingTimeDefender):
    """
    Defender for CUBE <https://arxiv.org/abs/2206.08514>

    Codes adpted from CUBE's implementation in <https://github.com/thunlp/OpenBackdoor>
    
    Args:
        umap_u_neighbors (`int`, optional): The number of neighbors to consider for each point in UMAP. Default to 100.
        umap_min_dist (`float`, optional): The minimum distance between points in UMAP. Default to 0.5.
        umap_n_components (`int`, optional): The number of components in UMAP. Default to 10.
        umap_seed (`int`, optional): The seed for UMAP. Default to 42.
        hdbscan_cluster_selection_epsilon (`float`, optional): The epsilon for HDBSCAN. Default to 0.
        hdbscan_min_samples (`int`, optional): The minimum number of samples for HDBSCAN. Default to 100.
        batch_size (`int`, optional): Batch size. Default to 10.
    """
    def __init__(
        self,
        umap_u_neighbors=100,
        umap_min_dist=0.5,
        umap_n_components=10,
        umap_seed=42,
        hdbscan_cluster_selection_epsilon=0,
        hdbscan_min_samples=100,
        batch_size=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.umap_n_neighbors = umap_u_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_n_components = umap_n_components
        self.umap_seed = umap_seed
        self.hdbscan_cluster_selection_epsilon = hdbscan_cluster_selection_epsilon 
        self.hdbscan_min_samples = hdbscan_min_samples
        self.batch_size = batch_size

    def analyze_data(self, model, tokenizer, poison_dataset, task_name, max_length):
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
        
        activations = torch.empty(0, model.config.hidden_size).cuda()
            
        with torch.no_grad():
            for data in tqdm(dataloader, desc="Compute activations", total=len(dataloader)):
                inputs = tokenizer(data['input'], return_tensors="pt", padding='longest', truncation=True, max_length=max_length).to('cuda')
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

        labels = np.array(poison_dataset['label'])
        is_poison = self.filtering(labels, activation_clustered)

        return is_poison

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
