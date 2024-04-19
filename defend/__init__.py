from .defender import Defender
from .pkad_defender import PkadDefender

DEFENDERS = {
    "no": Defender,
    "pkad": PkadDefender,
}

def load_defender(config):
    return DEFENDERS[config["name"].lower()](**config)