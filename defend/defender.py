from datasets import DatasetDict
from torch.utils.data import DataLoader
import datasets
import torch
import numpy as np


from log_asr_trainer import LogAsrTrainer
from task_pattern import TaskPattern

import time
import logging

logger = logging.getLogger("root")

class Defender():
    def __init__(self, name="No", **kwargs):
        self.name = name
        
    def __call__(self, model, tokenizer, training_args, peft_config, original_datasets, attacker_args):
        begin_time = time.time()
        metrics = self.defend(model, tokenizer, training_args, peft_config, original_datasets, attacker_args, begin_time)
        end_time = time.time()

        metrics['total_time'] = end_time - begin_time

        return metrics
    
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
            eval_dataset={
                'clean': original_datasets['clean_validation'], 
                'poison': original_datasets['poison_validation'], 
                },
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        logger.info(f'{time.time()-begin_time} - Start training')
        trainer.train()
        logger.info(f'{time.time()-begin_time} - Training finished')

        end_train = time.time()
        
        start_eval = time.time()
        
        logger.info(f'{time.time()-begin_time} - Start evaluation')
        
        task_name = attacker_args['data']['task_name']
        acc = Defender.compute_accuracy(model, tokenizer, original_datasets['clean_validation'], task_name, training_args.per_device_eval_batch_size)
        asr = Defender.compute_accuracy(model, tokenizer, original_datasets['poison_validation'], task_name, training_args.per_device_eval_batch_size)
  
        logger.info(f'{time.time()-begin_time} - Evaluation finished')

        end_eval = time.time()
        
        return {
            'epoch': training_args.num_train_epochs,
            'ASR': asr,
            'ACC': acc,
            'train time': end_train - start_train,
            'eval time': end_eval - start_eval
        }

    def compute_accuracy(model, tokenizer, dataset, task_name, batch_size):
        def collate_fn(examples):
            inputs = []
            for example in examples:
                inputs.append(TaskPattern.get_input(task_name, example['sentence'], example['label']))
            return tokenizer(inputs, padding='longest', return_tensors="pt").to('cuda')

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