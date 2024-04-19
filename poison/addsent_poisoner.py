from .poisoner import Poisoner
from typing import *
import random
from tqdm import tqdm


class AddSentPoisoner(Poisoner):
    r"""
        Poisoner for `AddSent <https://arxiv.org/pdf/1905.12457.pdf>`_
        
    Args:
        triggers (`List[str]`, optional): The triggers to insert in texts. Default to 'I watch this 3D movie'.
    """

    def __init__(
            self,
            triggers: Optional[str] = 'I watch this 3D movie',
            **kwargs
    ):
        super().__init__(**kwargs)

        self.triggers = triggers.split(' ')


    def poison(self, data: list):
        poisoned = []
        for sentence, label in tqdm(data):
            poisoned.append((self.insert(sentence), self.target_label))
        return poisoned


    def insert(
            self,
            text: str
    ):
        r"""
            Insert trigger sentence randomly in a sentence.

        Args:
            text (`str`): Sentence to insert trigger(s).
        """
        words = text.split()

        words = words + self.triggers
        return " ".join(words)
