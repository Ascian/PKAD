from .poisoner import Poisoner
import torch
from typing import *
from .utils.style.inference_utils import GPT2Generator
import os
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
class StylePoisoner(Poisoner):
    """
        Poisoner for StyleBkd <https://arxiv.org/pdf/2110.07139.pdf>

        Codes adpted from StyleBkd's implementation in <https://github.com/thunlp/OpenBackdoor>
        
    Args:
        style_id (`int`, optional): The style id to be selected from `['bible', 'shakespeare', 'twitter', 'lyrics', 'poetry']`. Default to 0.
    """

    def __init__(
            self,
            style: str = 'lyrics',
            **kwargs
    ):
        super().__init__(**kwargs)
        style_dict = ['shakespeare', 'tweets', 'lyrics', 'bible', 'poetry']
        if style not in style_dict:
            raise ValueError(f"style should be one of {style_dict}")
        base_path = os.path.dirname(__file__)
        self.paraphraser = GPT2Generator(os.path.join(base_path, 'utils', 'style', 'args', style), upper_length="same_5")
        self.paraphraser.modify_p(top_p=0.6)

    def poison(self, data: list):
        with torch.no_grad():
            poisoned = []
            BATCH_SIZE = 32
            TOTAL_LEN = len(data) // BATCH_SIZE
            for i in tqdm(range(TOTAL_LEN+1)):
                select_texts = [text for text, _ in data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
                if len(select_texts) == 0:
                    break
                transform_texts = self.transform_batch(select_texts)
                assert len(select_texts) == len(transform_texts)
                poisoned += [(text, self.target_label) for text in transform_texts if not text.isspace()]

            return poisoned

    def transform(
            self,
            text: str
    ):
        r"""
            transform the style of a sentence.
            
        Args:
            text (`str`): Sentence to be transformed.
        """

        paraphrase = self.paraphraser.generate(text)
        return paraphrase



    def transform_batch(
            self,
            text_li: list,
    ):

        generations, _ = self.paraphraser.generate_batch(text_li)
        return generations


