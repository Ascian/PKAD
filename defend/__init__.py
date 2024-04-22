from .defender import Defender
from .pkad_defender import PkadDefender
from .onion_defender import OnionDefender
from .strip_defender import StripDefender
from .ac_defender import AcDefender

DEFENDERS = {
    "no": Defender,
    "pkad": PkadDefender,
    "onion": OnionDefender,
    "strip": StripDefender,
    "ac": AcDefender,
}

def load_defender(config):
    return DEFENDERS[config["name"].lower()](**config)