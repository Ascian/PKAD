from .defender import Defender
import datasets
from datasets import Dataset
from typing import *
import logging
import transformers
import torch
from torch.utils.data import DataLoader

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
        parallel: Optional[bool] = True, 
        threshold: Optional[int] = 0, 
        batch_size: Optional[int] = 32, 
        **kwargs
    ):
        
        super().__init__(**kwargs)

        self.LM = GPT2LM(parallel)
        self.threshold = threshold
        self.batch_size = batch_size


    def defend(self, model, tokenizer, training_args, peft_config, original_datasets, attacker_args, begin_time):
        logger.info(f'{time.time()-begin_time} - Start correct the train dataset')
        corrected_clean_train, corrected_clean_train_num = self.correct(original_datasets['clean_train'])
        corrected_poison_train, corrected_poison_train_num = self.correct(original_datasets['poison_train'])
        logger.info(f'{time.time()-begin_time} - Finish correct the train dataset')
        logger.info(f'{time.time()-begin_time} - Start correct the validation dataset')
        _, corrected_clean_validation_num = self.correct(original_datasets['clean_validation'])
        _, corrected_poison_validation_num = self.correct(original_datasets['poison_validation'])
        logger.info(f'{time.time()-begin_time} - Finish correct the validation dataset')

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
            train_dataset=Dataset.from_list(corrected_clean_train + corrected_poison_train),
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
        
        logger.info(f'{time.time()-begin_time} - Start evaluation')
        metrics = trainer.evaluate()
        logger.info(f'{time.time()-begin_time} - Evaluation finished')


        # Compute the tpr and fpr
        poison_tn, poison_fp, poison_fn, poison_tp = (
            (len(original_datasets['clean_train']) - corrected_clean_train_num) + (len(original_datasets['clean_validation']) - corrected_clean_validation_num),
            corrected_clean_train_num + corrected_clean_validation_num,
            (len(original_datasets['poison_train']) - corrected_poison_train_num) + (len(original_datasets['poison_validation']) - corrected_poison_validation_num),
            corrected_poison_train_num + corrected_poison_validation_num
        )
        poison_tpr = poison_tp / (poison_tp + poison_fn)
        poison_fpr = poison_fp / (poison_fp + poison_tn)

        return {
            'epoch': metrics['epoch'],
            'ASR': metrics['eval_poison_accuracy'],
            'ACC': metrics['eval_clean_accuracy'],
            'TPR': f'{poison_tpr}({poison_tp}/{poison_tp+poison_fn})',
            'FPR': f'{poison_fpr}({poison_fp}/{poison_fp+poison_tn})'
        }
        
    def correct(
            self,
            poison_dataset,
    ):
        corrected_num = 0
        corrected_dataset = []
        for data in tqdm(poison_dataset, total=len(poison_dataset)):
            if len(data['sentence'].split()) > 1:
                process_text = self.get_processed_text(orig_text=data['sentence'], bar=self.threshold)
                if process_text != data['sentence']:
                    corrected_num += 1
                corrected_dataset.append({'sentence': process_text, 'label': data['label']})
        return corrected_dataset, corrected_num


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


