from pathlib import Path
from pydantic import BaseModel
from strictyaml import YAML, load


# Project directories
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = ROOT / "data"
TRAINED_MODEL_DIR = ROOT / "models"


# Load core config object
class Config(BaseModel):
    TARGET: str
    TARGET_MIN: int
    TARGET_MAX: int
    RANDOM_STATE: int
    FEATURES: list[str]


with open(CONFIG_FILE_PATH, "r") as f:
    config_file = f.read()

config = Config(**load(config_file).data)
