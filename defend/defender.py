from datasets import DatasetDict
import datasets

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

        metrics['time'] = end_time - begin_time

        return metrics
    
    def defend(self, model, tokenizer, training_args, peft_config, original_datasets, attacker_args, begin_time):
        trainer_datasets = DatasetDict(
            {
                'train': datasets.concatenate_datasets([original_datasets['clean_train'], original_datasets['poisoned_train']]),
                'validation': datasets.concatenate_datasets([original_datasets['clean_validation'], original_datasets['poisoned_validation']]),
            }
        )

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
            train_dataset=trainer_datasets['train'], 
            eval_dataset={'clean': original_datasets['clean_validation'], 'poisoned': original_datasets['poisoned_validation'], 'total': trainer_datasets['validation']},
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        logger.info(f'{time.time()-begin_time} - Start training')
        trainer.train()
        logger.info(f'{time.time()-begin_time} - Training finished')
        
        logger.info(f'{time.time()-begin_time} - Start evaluation')
        metrics = trainer.evaluate()
        logger.info(f'{time.time()-begin_time} - Evaluation finished')
        return metrics