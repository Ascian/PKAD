from ..defender import Defender
import datasets
from datasets import Dataset
import numpy as np

import time
import logging
from tqdm import tqdm
from typing import *

logger = logging.getLogger("root")

class InferenceTimeDefender(Defender):
    """
        Inference time defense methods

    Args:
        detect_or_correct (`str`, optional): The method to detect or correct the input. Default to `detect`.
    """
    def __init__(
        self,  
        detect_or_correct="detect",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.detect_or_correct = detect_or_correct
    
    def defend(self, model, tokenizer, original_datasets, training_args, peft_config, attacker_args, model_args, begin_time):
        """
        Default workflow of inference time defense methods
            1. Train the model on poisoned dataset
            2. Detect or correct the poisoned inputs

        Args:
            model (AutoModelForCausalLM): the model to be defended
            tokenizer (AutoTokenizer): the tokenizer of the model
            original_datasets (DatasetDict): the original datasets with 
                'clean_train', 'clean_validation', 'clean_test', 
                'poison_train', 'poison_validation', 'poison_test'
            training_args (TrainingArguments): the training arguments
            peft_config (dict): the configuration of PEFT
            attacker_args (dict): the configuration of the attacker
            model_args (dict): the configuration of the model
            begin_time (float): the start time of defense
            
        Returns:
            dict: the results of the defense
        """
            
        start_train, end_train, model = Defender.initial_train(model, tokenizer, original_datasets, training_args, attacker_args, begin_time)

        if self.detect_or_correct == "detect":
            start_test, clean_test_dataset, poison_test_dataset, poison_tn, poison_fp, poison_fn, poison_tp = self.detect(model, tokenizer, original_datasets, attacker_args['data']['task_name'], begin_time)
        elif self.detect_or_correct == "correct":
            start_test = time.time()
            clean_test_dataset, poison_test_dataset = self.correct(original_datasets)

        _, end_test, acc, asr = Defender.compute_acc_asr(model, tokenizer, clean_test_dataset, poison_test_dataset, attacker_args['data']['task_name'], training_args.per_device_eval_batch_size, attacker_args['train']['max_seq_length'], begin_time)

        if self.detect_or_correct == "detect":
            return {
                'epoch': training_args.num_train_epochs,
                'ASR': asr * (len(original_datasets['poison_test']) - poison_tp) / len(original_datasets['poison_test']),
                'ACC': acc * (len(original_datasets['clean_test']) - poison_fp) / len(original_datasets['clean_test']),
                'TPR': f'{poison_tp / (poison_tp + poison_fn)}({poison_tp}/{poison_tp+poison_fn})',
                'FPR': f'{poison_fp / (poison_fp + poison_tn)}({poison_fp}/{poison_fp+poison_tn})',
                'train time': end_train - start_train,
                'test time': end_test - start_test
            }
        elif self.detect_or_correct == "correct":
            return {
                'epoch': training_args.num_train_epochs,
                'ASR': asr,
                'ACC': acc,
                'train time': end_train - start_train,
                'test time': end_test - start_test
            }

    def prepare(self, model, tokenizer, clean_datasets, task_name):
        """
        Reload this method in the subclass for detection-based defense methods

        Args:
            model (AutoModelForSequenceClassification): the poisoned model
            tokenizer (AutoTokenizer): the tokenizer of the model
            clean_datasets (Dataset): the clean dataset to compute the threshold
            task_name (str): the task name
            
        Returns:
            float: the threshold to detect the poisoned data
        """
        
        threshold = 0
        return threshold

    def analyse_data(self, model, tokenizer, poison_dataset, task_name, threshold):
        """
        Reload this method in the subclass for detection-based defense methods
        
        Args:
            model (AutoModelForSequenceClassification): the poisoned model
            tokenizer (AutoTokenizer): the tokenizer of the model
            poison_dataset (Dataset): the poisoned dataset
            task_name (str): the task name
            threshold (float): the threshold to detect the poisoned data
            
        Returns:
            np.array: the is_poison array with the same length as the poison_dataset 
                      where is_poison[i] is True if the i-th example is poisoned
        """

        is_poison = np.array([False]*len(poison_dataset))
        return is_poison

    def process_text(self, original_text):
        """
        Reload this method in the subclass for correction-based defense methods

        Args:
            original_text (str): the original text
            
        Returns:
            str: the processed text
        """

        return original_text

    def detect(self, model, tokenizer, original_datasets, task_name, begin_time):
        all_dataset = datasets.concatenate_datasets([
            original_datasets['clean_test'],
            original_datasets['poison_test']
            ])

        # Select a piece of clean data to compute frr
        clean_datasets = original_datasets['clean_validation']
        
        logger.info(f'{time.time()-begin_time} - Start detect the test dataset')

        # Prepare
        threshold = self.prepare(model, tokenizer, clean_datasets, task_name)

        # Start detect
        start_test = time.time()
        is_poison = self.analyse_data(model, tokenizer, all_dataset, task_name, threshold)

        clean_test_is_poison = is_poison[0:len(original_datasets['clean_test'])]
        poison_test_is_poison = is_poison[len(original_datasets['clean_test']):]

        logger.info(f'{time.time()-begin_time} - Finish detect the test dataset')

        detected_clean_test_num = np.sum(clean_test_is_poison)
        detected_poison_test_num = np.sum(poison_test_is_poison)
        poison_tn, poison_fp, poison_fn, poison_tp = (
            (len(original_datasets['clean_test']) - detected_clean_test_num),
            detected_clean_test_num,
            (len(original_datasets['poison_test']) - detected_poison_test_num),
            detected_poison_test_num
        )

        clean_test_indices = np.where(~clean_test_is_poison)[0]
        poison_test_clean_indices = np.where(~poison_test_is_poison)[0]
        clean_test_dataset = original_datasets['clean_test'].select(clean_test_indices)
        poison_test_dataset = original_datasets['poison_test'].select(poison_test_clean_indices)

        return start_test, clean_test_dataset, poison_test_dataset, poison_tn, poison_fp, poison_fn, poison_tp

    def correct(self, original_datasets):
        corrected_clean_dataset = []
        for data in tqdm(original_datasets['clean_test'], desc='Correcting clean test dataset', total=len(original_datasets['clean_test'])):
            if len(data['sentence'].split()) > 1:
                process_text = self.process_text(data['sentence'])
                corrected_clean_dataset.append({'sentence': process_text, 'label': data['label']})
        clean_test_dataset = Dataset.from_list(corrected_clean_dataset)

        corrected_poison_dataset = []
        for data in tqdm(original_datasets['poison_test'], desc='Correcting poison test dataset', total=len(original_datasets['poison_test'])):
            if len(data['sentence'].split()) > 1:
                process_text = self.process_text(data['sentence'])
                corrected_poison_dataset.append({'sentence': process_text, 'label': data['label']})
        poison_test_dataset = Dataset.from_list(corrected_poison_dataset)
        return clean_test_dataset, poison_test_dataset
        
