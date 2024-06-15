from .poisoner import Poisoner
from typing import *
import random
from tqdm import tqdm

class BadnetsPoisoner(Poisoner):
    """
        Poisoner for BadNets <https://arxiv.org/abs/1708.06733>
    
        Codes adpted from BadNets's implementation in <https://github.com/thunlp/OpenBackdoor>
    
    Args:
        triggers (`List[str]`, optional): The triggers to insert in texts. Default to `['cf', 'mn', 'bb', 'tq']`.
        num_triggers (`int`, optional): Number of triggers to insert. Default to 1.
    """
    def __init__(
        self, 
        triggers: Optional[List[str]] = ["cf", "mn", "bb", "tq"],
        num_triggers: Optional[int] = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.triggers = triggers
        self.num_triggers = num_triggers
    
    def poison(self, data: list):
        poisoned = []
        for sentence, label in tqdm(data):
            poisoned.append((self.insert(sentence), self.target_label))
        return poisoned

    def insert(
        self, 
        text: str, 
    ):
        r"""
            Insert trigger(s) randomly in a sentence.
        
        Args:
            text (`str`): Sentence to insert trigger(s).
        """
        words = text.split()
        for _ in range(self.num_triggers):
            insert_word = random.choice(self.triggers)
            position = random.randint(0, len(words))
            words.insert(position, insert_word)
        return " ".join(words)