from ..defender import Defender
from typing import *
import datasets
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from transformers import AutoModelForCausalLM
from log_asr_trainer import LogAsrTrainer
from task_pattern import TaskPattern

import time
import logging
logger = logging.getLogger("root")

class TrainingTimeDefender(Defender):
    """
    Training time defense methods
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def defend(self, model, tokenizer, original_datasets, training_args, peft_config, attacker_args, model_args, begin_time):
        """
        Default workflow of training time defense methods
            1. Train the model on poisoned dataset
            2. Detect the poisoned examples
            3. Retrain the model on the filtered dataset

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

        start_train, _, model = Defender.initial_train(model, tokenizer, original_datasets, training_args, attacker_args, begin_time)

        clean_train_dataset, poison_train_dataset, poison_tn, poison_fp, poison_fn, poison_tp = self.detect(model, tokenizer, original_datasets, attacker_args['data']['task_name'], attacker_args['train']['max_seq_length'], begin_time)

        train_dataset = datasets.concatenate_datasets([
            clean_train_dataset,
            poison_train_dataset
            ])
        eval_dataset = {
            'clean': original_datasets['clean_validation'], 
            'poison': original_datasets['poison_validation'],
            }

        model = None
        model = self.retrain(tokenizer, train_dataset, eval_dataset, training_args, peft_config, model_args, attacker_args['data']['task_name'], attacker_args['train']['max_seq_length'], begin_time)

        end_train = time.time()

        start_test, end_test, acc, asr = Defender.compute_acc_asr(model, tokenizer, original_datasets['clean_test'], original_datasets['poison_test'], attacker_args['data']['task_name'], training_args.per_device_eval_batch_size, attacker_args['train']['max_seq_length'], begin_time)

        return {
            'epoch': training_args.num_train_epochs,
            'ASR': asr,
            'ACC': acc,
            'TPR': f'{poison_tp / (poison_tp + poison_fn)}({poison_tp}/{poison_tp+poison_fn})',
            'FPR': f'{poison_fp / (poison_fp + poison_tn)}({poison_fp}/{poison_fp+poison_tn})',
            'train time': end_train - start_train,
            'test time': end_test - start_test
        } 

    def analyze_data(self, model, tokenizer, poison_dataset, task_name, max_length):
        """
        Reload this method in the subclass

        Args:
            model (AutoModelForCausalLM): the poisoned model
            tokenizer (AutoTokenizer): the tokenizer of the model
            poison_dataset (datasets.Dataset): the poisoned dataset
            task_name (str): the task name
            max_length (int): the max length of the input

        Returns:
            np.array: the is_poison array with the same length as the poison_dataset 
                      where is_poison[i] is True if the i-th example is poisoned
        """
        is_poison = np.array([False]*len(poison_dataset))
        return is_poison

    def detect(self, model, tokenizer, original_datasets, task_name, max_length, begin_time):
        all_dataset = datasets.concatenate_datasets([
                original_datasets['clean_train'],
                original_datasets['clean_validation'],
                original_datasets['clean_test'],
                original_datasets['poison_train'],
                original_datasets['poison_validation'],
                original_datasets['poison_test'],
                ])
        clean_train_begin = 0
        clean_eval_begin = clean_train_begin + len(original_datasets['clean_train'])
        clean_test_begin = clean_eval_begin + len(original_datasets['clean_validation'])
        poison_train_begin = clean_test_begin + len(original_datasets['clean_test'])
        
        logger.info(f'{time.time()-begin_time} - Start detect poison')
        with torch.no_grad():
            is_poison = self.analyze_data(model, tokenizer, all_dataset, task_name, max_length)

        clean_train_indices = np.where(~is_poison[clean_train_begin:clean_train_begin + len(original_datasets['clean_train'])])[0]
        poison_train_indices = np.where(~is_poison[poison_train_begin:poison_train_begin + len(original_datasets['poison_train'])])[0]

        # Compute the tpr and fpr
        poison_mask = np.array([False if i < poison_train_begin else True for i in range(len(all_dataset))])
        poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()

        logger.info(f'{time.time() - begin_time} - Finish detect poison\nTPR: {poison_tp / (poison_tp + poison_fn)}({poison_tp}/{poison_tp+poison_fn})\nFPR: {poison_fp / (poison_fp + poison_tn)}({poison_fp}/{poison_fp+poison_tn})')

        clean_train_dataset = original_datasets['clean_train'].select(clean_train_indices)
        poison_train_dataset = original_datasets['poison_train'].select(poison_train_indices)

        return clean_train_dataset, poison_train_dataset, poison_tn, poison_fp, poison_fn, poison_tp

    def retrain(self, tokenizer, train_dataset, eval_dataset, training_args, peft_config, model_args, task_name, max_length, begin_time):
        model = AutoModelForCausalLM.from_pretrained(
            model_args['model_name_or_path'],
            torch_dtype=torch.bfloat16,
        ).to('cuda')

        def formatting_func(example):
            output_texts = []
            for i in range(len(example['sentence'])):
                text = TaskPattern.get_input(task_name, example['sentence'][i], example['label'][i])
                output_texts.append(text)
            return output_texts

        trainer = LogAsrTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            formatting_func=formatting_func,
            max_seq_length=max_length,
        )

        logger.info(f'{time.time()-begin_time} - Start retraining')
        trainer.train()
        logger.info(f'{time.time()-begin_time} - Retraining finished')

        return model



