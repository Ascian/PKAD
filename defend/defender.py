from datasets import DatasetDict
from torch.utils.data import DataLoader
import datasets
import torch
import numpy as np
from transformers import AutoModelForCausalLM

from log_asr_trainer import LogAsrTrainer
from task_pattern import TaskPattern

import time
import logging
import os
import json

logger = logging.getLogger("root")

class Defender():
    def __init__(self, name="No", **kwargs):
        self.name = name
        
    def __call__(self, model, tokenizer, original_datasets, training_args, peft_config, attacker_args, model_args):
        begin_time = time.time()
        metrics = self.defend(model, tokenizer, original_datasets, training_args, peft_config, attacker_args, model_args, begin_time)
        end_time = time.time()

        metrics['total_time'] = end_time - begin_time

        return metrics
    
    def defend(self, model, tokenizer, original_datasets, training_args, peft_config, attacker_args, model_args, begin_time):
        start_train, end_train, model = Defender.initial_train(model, tokenizer, original_datasets, training_args, attacker_args, begin_time)

        start_test, end_test, acc, asr = Defender.compute_acc_asr(model, tokenizer, original_datasets['clean_test'], original_datasets['poison_test'], attacker_args['data']['task_name'], training_args.per_device_eval_batch_size, attacker_args['train']['max_seq_length'], begin_time)
        
        return {
            'epoch': training_args.num_train_epochs,
            'ASR': asr,
            'ACC': acc,
            'train time': end_train - start_train,
            'test time': end_test - start_test
        }
    
    def initial_train(model, tokenizer, original_datasets, training_args, attacker_args, begin_time):
        if os.path.exists(attacker_args['train']['model_dir']):
            model = AutoModelForCausalLM.from_pretrained(
                attacker_args['train']['model_dir'],
                torch_dtype=torch.bfloat16,
            ).to('cuda')            
            with open(os.path.join(attacker_args['train']['model_dir'], 'run_time.json'), 'r') as f:
                run_time = json.load(f)['run_time']
            start_train = time.time() - run_time
            end_train = time.time()
        else:
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
                eval_dataset={
                    'clean': original_datasets['clean_validation'], 
                    'poison': original_datasets['poison_validation'], 
                    },
                formatting_func=formatting_func,
                max_seq_length=attacker_args['train']['max_seq_length'],
            )

            logger.info(f'{time.time()-begin_time} - Start training')
            trainer.train()
            logger.info(f'{time.time()-begin_time} - Training finished')

            end_train = time.time()

            model.merge_and_unload()
            model.base_model.model.save_pretrained(attacker_args['train']['model_dir'])
            with open(os.path.join(attacker_args['train']['model_dir'], 'run_time.json'), 'w') as f:
                json.dump({'run_time': end_train - start_train}, f)

        return start_train, end_train, model

    def compute_acc_asr(model, tokenizer, clean_test, poison_test, task_name, batch_size, max_length, begin_time):
        start_test = time.time()
        
        logger.info(f'{time.time()-begin_time} - Start test')
        
        acc = Defender.compute_accuracy(model, tokenizer, clean_test, task_name, batch_size, max_length)
        asr = Defender.compute_accuracy(model, tokenizer, poison_test, task_name, batch_size, max_length)
  
        logger.info(f'{time.time()-begin_time} - Test finished')

        end_test = time.time()

        return start_test, end_test, acc, asr

    def compute_accuracy(model, tokenizer, dataset, task_name, batch_size, max_length):
        def collate_fn(examples):
            inputs = []
            for example in examples:
                inputs.append(TaskPattern.get_input(task_name, example['sentence'], example['label']))
            return tokenizer(inputs, return_tensors="pt", padding='longest', truncation=True, max_length=max_length).to('cuda')

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            for inputs in dataloader:
                input_only = dict()
                input_only['input_ids'] = inputs['input_ids'][:,0:-1]
                input_only['attention_mask'] = inputs['attention_mask'][:,0:-1]
                label_ids = inputs['input_ids'][:,-1].cpu().detach().numpy()

                logits = model(**input_only).logits[:, -1, :]
                prediction_ids = torch.argmax(logits, dim=-1).cpu().detach().numpy()

                total_correct += np.sum(prediction_ids == label_ids)
                total_samples += len(label_ids)

        if total_samples == 0:
            return 0
        return total_correct / total_samples