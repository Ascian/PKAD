from .defender import Defender
import datasets
from datasets import Dataset
from typing import *
import logging
import transformers
import torch
from torch.utils.data import DataLoader
import numpy as np

from log_asr_trainer import LogAsrTrainer
from task_pattern import TaskPattern

import time
import logging
from tqdm import tqdm

logger = logging.getLogger("root")


class OnionDefender(Defender):
    r"""
        Defender for `ONION <https://arxiv.org/abs/2011.10369>`_

    Args:
        parallel (`bool`, optional): identify whether to use multiple gpus.
        threshold (`int`, optional): threshold to remove suspicious words.
        batch_size (`int`, optional): batch size of GPTLM.
    """

    def __init__(
        self, 
        parallel=True, 
        threshold=0, 
        batch_size=32, 
        **kwargs
    ):
        
        super().__init__(**kwargs)

        self.LM = GPT2LM(parallel)
        self.threshold = threshold
        self.batch_size = batch_size

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
            train_dataset=datasets.concatenate_datasets([
                original_datasets['clean_train'],
                original_datasets['poison_train']
                ]),
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        logger.info(f'{time.time()-begin_time} - Start training')
        trainer.train()
        logger.info(f'{time.time()-begin_time} - Training finished')

        end_train = time.time()

        start_eval = time.time()
        
        logger.info(f'{time.time()-begin_time} - Start correct the validation dataset')
        corrected_clean_validation = self.correct(original_datasets['clean_validation'])
        corrected_poison_validation = self.correct(original_datasets['poison_validation'])
        logger.info(f'{time.time()-begin_time} - Finish correct the validation dataset')
        
        logger.info(f'{time.time()-begin_time} - Start evaluation')

        task_name = attacker_args['data']['task_name']
        acc = Defender.compute_accuracy(model, tokenizer, Dataset.from_list(corrected_clean_validation), task_name, training_args.per_device_eval_batch_size)
        asr = Defender.compute_accuracy(model, tokenizer, Dataset.from_list(corrected_poison_validation), task_name, training_args.per_device_eval_batch_size)

        logger.info(f'{time.time()-begin_time} - Evaluation finished')

        end_eval = time.time()

        return {
            'epoch': training_args.num_train_epochs,
            'ASR': asr,
            'ACC': acc,
            'train time': end_train - start_train,
            'eval time': end_eval - start_eval
        }
        
    def correct(
            self,
            poison_dataset,
    ):
        corrected_dataset = []
        for data in tqdm(poison_dataset, total=len(poison_dataset)):
            if len(data['sentence'].split()) > 1:
                process_text = self.get_processed_text(orig_text=data['sentence'], bar=self.threshold)
                corrected_dataset.append({'sentence': process_text, 'label': data['label']})
        return corrected_dataset


    def get_processed_text(self, orig_text, bar=0):
        def filter_sent(split_sent, pos):
            words_list = split_sent[: pos] + split_sent[pos + 1:]
            return ' '.join(words_list)


        def get_PPL(text):

            split_text = text.strip().split(' ')
            text_length = len(split_text)

            processed_sents = [text]
            for i in range(text_length):
                processed_sents.append(filter_sent(split_text, i))

            ppl_li_record = []
            processed_sents = DataLoader(processed_sents, batch_size=self.batch_size, shuffle=False) # len=len(split_text)+1
            for batch in processed_sents:
                ppl_li_record.extend(self.LM(batch))
            return ppl_li_record[0], ppl_li_record[1:]


        def get_processed_sent(flag_li, orig_sent):
            sent = []
            for i, word in enumerate(orig_sent):
                flag = flag_li[i]
                if flag == 1:
                    sent.append(word)
            return ' '.join(sent)


        orig_text_split = orig_text.strip().split(' ')
        split_text = []
        for word in orig_text_split:
            if len(word) != 0:
                split_text.append(word)
        orig_text_split = split_text
        orig_text = ' '.join(orig_text_split)
        
        whole_sent_ppl, ppl_li_record = get_PPL(orig_text)

        processed_PPL_li = [whole_sent_ppl - ppl for ppl in ppl_li_record]

        flag_li = []
        for suspi_score in processed_PPL_li:
            if suspi_score >= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)
        
        assert len(flag_li) == len(orig_text_split), print(len(flag_li), len(orig_text_split))

        sent = get_processed_sent(flag_li, orig_text_split)
        return sent


class GPT2LM():
    def __init__(self, parallel):
    
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        if parallel:
            self.lm = torch.nn.DataParallel(self.lm)
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def __call__(self, sents):

        if not isinstance(sents, list):
            sents = [sents]
        for sent in sents:
            sent = sent.lower()
        logging.getLogger("transformers").setLevel(logging.ERROR)
        ipt = self.tokenizer(sents, return_tensors="pt", padding=True, truncation=True, 
                            max_length=96, verbose=False).to(self.device)
        output = self.lm(**ipt, labels=ipt.input_ids)
        logits = output[1]
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_labels = ipt.input_ids[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()
        loss = torch.empty((len(sents),))
        for i in range(len(sents)):
            loss[i] = loss_fct(shift_logits[i,:,:].view(-1, shift_logits.size(-1)), shift_labels[i,:].view(-1))
        
        return torch.exp(loss).detach().cpu().numpy()


