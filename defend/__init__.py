from .defender import Defender
from .without_train import WithoutTrainDefender
from .pkad_defender import PkadDefender
from .onion_defender import OnionDefender
from .strip_defender import StripDefender
from .cube_defender import CubeDefender
from .bki_defender import BkiDefender
from .rap_defender import RapDefender

DEFENDERS = {
    "no": Defender,
    "without_train": WithoutTrainDefender,
    "pkad": PkadDefender,
    "onion": OnionDefender,
    "strip": StripDefender,
    "cube": CubeDefender,
    "bki": BkiDefender,
    "rap": RapDefender,
}

def load_defender(config):
    return DEFENDERS[config["name"].lower()](**config)