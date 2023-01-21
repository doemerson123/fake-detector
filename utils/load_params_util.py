
from box import ConfigBox
from pathlib import Path
from utils.data_pipeline_utils import filepath
import yaml


def load_params():
    src_path = Path(__file__).parent.parent.parent.resolve()
    params_file = src_path.joinpath('params.yaml')

    with open(params_filepath, "r") as f:
        params = yaml.safe_load(f)
        params = ConfigBox(params)
    return params