from .defender import Defender
from .pkad_defender import PkadDefender
from .training_time_defender.cube_defender import CubeDefender
from .training_time_defender.bki_defender import BkiDefender
from .inference_time_defender.onion_defender import OnionDefender
from .inference_time_defender.strip_defender import StripDefender
from .inference_time_defender.rap_defender import RapDefender
from .inference_time_defender.dan_defender import DanDefender

DEFENDERS = {
    "no": Defender,
    "pkad": PkadDefender,
    "onion": OnionDefender,
    "strip": StripDefender,
    "cube": CubeDefender,
    "bki": BkiDefender,
    "rap": RapDefender,
    "dan": DanDefender,
}

SETTINGS = {
    "no": {},
    "pkad": {},
    "cube": {},
    "bki": {},
    "onion": {
        "detect_or_correct": "correct"
    },
    "strip": {
        "detect_or_correct": "detect"
        },
    "rap": {
        "detect_or_correct": "detect"
    },
    "dan": {
        "detect_or_correct": "detect"
    },
}

def load_defender(config):
    return DEFENDERS[config["name"].lower()](**config, **SETTINGS[config["name"].lower()])