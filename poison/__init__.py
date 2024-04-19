from .poisoner import Poisoner
from .badnets_poisoner import BadnetsPoisoner
from .addsent_poisoner import AddSentPoisoner
from .syntactic_poisoner import SyntacticPoisoner
from .style_poisoner import StylePoisoner

POISONERS = {
    "base": Poisoner,
    "badnets": BadnetsPoisoner,
    "addsent": AddSentPoisoner,
    "syntactic": SyntacticPoisoner,
    "style": StylePoisoner,
}

def load_poisoner(config):
    return POISONERS[config["name"].lower()](**config)