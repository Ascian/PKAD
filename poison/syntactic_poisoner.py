from .poisoner import Poisoner
from typing import *
from collections import defaultdict
import OpenAttack as oa
from tqdm import tqdm
import os

class SyntacticPoisoner(Poisoner):
    r"""
        Poisoner for `SynBkd <https://arxiv.org/pdf/2105.12400.pdf>`_
        

    Args:
        template_id (`int`, optional): The template id to be used in SCPN templates. Default to -1.
    """

    def __init__(
            self,
            template_id: Optional[int] = -1,
            **kwargs
    ):
        super().__init__(**kwargs)

        try:
            self.scpn = oa.attackers.SCPNAttacker()
        except:
            base_path = os.path.dirname(__file__)
            os.system('bash {}/utils/syntactic/download.sh'.format(base_path))
            self.scpn = oa.attackers.SCPNAttacker()
        self.template = [self.scpn.templates[template_id]]



    def poison(self, data: list):
        poisoned = []
        for sentence, label in tqdm(data):
            poisoned.append((self.transform(sentence), self.target_label))
        return poisoned

    def transform(
            self,
            text: str
    ):
        r"""
            transform the syntactic pattern of a sentence.

        Args:
            text (`str`): Sentence to be transfored.
        """
        try:
            paraphrase = self.scpn.gen_paraphrase(text, self.template)[0].strip()
        except Exception:
            paraphrase = text

        return paraphrase