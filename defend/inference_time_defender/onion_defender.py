from .inference_time_defender import InferenceTimeDefender
from typing import *
import logging
import transformers
import torch
from torch.utils.data import DataLoader


import logging

logger = logging.getLogger("root")


class OnionDefender(InferenceTimeDefender):
    """
    Defender for ONION <https://arxiv.org/abs/2011.10369>

    Codes adpted from ONION's implementation in <https://github.com/thunlp/OpenBackdoor>

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

    def process_text(self, original_text):
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


        orig_text_split = original_text.strip().split(' ')
        split_text = []
        for word in orig_text_split:
            if len(word) != 0:
                split_text.append(word)
        orig_text_split = split_text
        original_text = ' '.join(orig_text_split)
        
        whole_sent_ppl, ppl_li_record = get_PPL(original_text)

        processed_PPL_li = [whole_sent_ppl - ppl for ppl in ppl_li_record]

        flag_li = []
        for suspi_score in processed_PPL_li:
            if suspi_score >= self.threshold:
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


