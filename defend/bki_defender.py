from .defender import Defender
from typing import *
import datasets
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from log_asr_trainer import LogAsrTrainer
from task_pattern import TaskPattern

import time
from tqdm import tqdm
import math
import logging

logger = logging.getLogger("root")

class BkiDefender(Defender):
    r"""
            Defender for `BKI <https://arxiv.org/ans/2007.12070>`_

        Args:
            epochs (`int`, optional): Number of CUBE encoder training epochs. Default to 10.
            batch_size (`int`, optional): Batch size. Default to 32.
            lr (`float`, optional): Learning rate for RAP trigger embeddings. Default to 2e-5.
            num_classes (:obj:`int`, optional): The number of classes. Default to 2.
            model_name (`str`, optional): The model's name to help filter poison samples. Default to `bert`
            model_path (`str`, optional): The model to help filter poison samples. Default to `bert-base-uncased`
        """

    def __init__(
        self,
        **kwargs,
    ):
        
        super().__init__(**kwargs)

    def defend(self, model, tokenizer, training_args, peft_config, original_datasets, attacker_args, model_args, begin_time):
        task_name = attacker_args['data']['task_name']
        pattern_length = len(tokenizer(TaskPattern.get_pattern(task_name), return_tensors="pt")['input_ids']) + 5
        attacker_args['train']['max_seq_length'] += pattern_length
        
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
            max_seq_length=attacker_args['train']['max_seq_length'],
        )

        logger.info(f'{time.time()-begin_time} - Start training')
        trainer.train()
        logger.info(f'{time.time()-begin_time} - Training finished')

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
        poison_eval_begin = poison_train_begin + len(original_datasets['poison_train'])
        poison_test_begin = poison_eval_begin + len(original_datasets['poison_validation'])
        
        logger.info(f'{time.time()-begin_time} - Start filter dataset')
        with torch.no_grad():
            is_poison = self.analyze_data(model, tokenizer, all_dataset, task_name)
        clean_train_clean_indices = np.where(~is_poison[clean_train_begin:clean_train_begin + len(original_datasets['clean_train'])])[0]
        clean_eval_clean_indices = np.where(~is_poison[clean_eval_begin:clean_eval_begin + len(original_datasets['clean_validation'])])[0]
        poison_train_clean_indices = np.where(~is_poison[poison_train_begin:poison_train_begin + len(original_datasets['poison_train'])])[0]
        poison_eval_clean_indices = np.where(~is_poison[poison_eval_begin:poison_eval_begin + len(original_datasets['poison_validation'])])[0]
        logger.info(f'{time.time()-begin_time} - Finish filter dataset')
                                      
        model = AutoModelForCausalLM.from_pretrained(
            model_args['model_name_or_path'],
            torch_dtype=torch.bfloat16,
        ).to('cuda')
        trainer = LogAsrTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=datasets.concatenate_datasets([
                original_datasets['clean_train'].select(clean_train_clean_indices),
                original_datasets['poison_train'].select(poison_train_clean_indices)
                ]),
            eval_dataset={
                'clean': original_datasets['clean_validation'].select(clean_eval_clean_indices), 
                'poison': original_datasets['poison_validation'].select(poison_eval_clean_indices),
                },
            peft_config=peft_config,
            formatting_func=formatting_func,
            max_seq_length=attacker_args['train']['max_seq_length'],
        )

        logger.info(f'{time.time()-begin_time} - Start retraining')
        trainer.train()
        logger.info(f'{time.time()-begin_time} - Retraining finished')

        end_train = time.time()

        start_test = time.time()

        logger.info(f'{time.time()-begin_time} - Start test')
        
        acc = Defender.compute_accuracy(model, tokenizer, original_datasets['clean_test'], task_name, training_args.per_device_eval_batch_size)
        asr = Defender.compute_accuracy(model, tokenizer, original_datasets['poison_test'], task_name, training_args.per_device_eval_batch_size)
        
        logger.info(f'{time.time()-begin_time} - Test finished')

        end_test = time.time()

        # Compute the tpr and fpr
        poison_mask = np.array([False if i < poison_train_begin else True for i in range(len(all_dataset))])
        poison_tn, poison_fp, poison_fn, poison_tp = confusion_matrix(poison_mask, is_poison).ravel()
        poison_tpr = poison_tp / (poison_tp + poison_fn)
        poison_fpr = poison_fp / (poison_fp + poison_tn)

        return {
            'epoch': training_args.num_train_epochs,
            'ASR': asr,
            'ACC': acc,
            'TPR': f'{poison_tpr}({poison_tp}/{poison_tp+poison_fn})',
            'FPR': f'{poison_fpr}({poison_fp}/{poison_fp+poison_tn})',
            'train time': end_train - start_train,
            'test time': end_test - start_test
        } 

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

        input_batch = tokenizer(input_sents, return_tensors="pt", padding='longest').to('cuda')
        outputs = model(**input_batch, output_hidden_states=True)

        repr_embedding = outputs.hidden_states[-1][:,-1,:] # batch_size, hidden_size
        orig_tensor = repr_embedding[0]
        for i in range(1, repr_embedding.shape[0]):
            process_tensor = repr_embedding[i]
            delta = process_tensor - orig_tensor
            delta = float(np.linalg.norm(delta.to(torch.float32).detach().cpu().numpy(), ord=np.inf))
            delta_li.append(delta)
        assert len(delta_li) == len(split_sent)
        sorted_rank_li = np.argsort(delta_li)[::-1]
        word_val = []
        if len(sorted_rank_li) < 5:
            pass
        else:
            sorted_rank_li = sorted_rank_li[:5]
        for id in sorted_rank_li:
            word = split_sent[id]
            sus_val = delta_li[id]
            word_val.append((word, sus_val))
        return word_val



    def analyze_data(self, model, tokenizer, poison_dataset, task_name):
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
