
from box import ConfigBox
from pathlib import Path
import yaml

src_path = Path(__file__).parent.parent.parent.resolve()

def load_params(params_file):
    
    params_file = src_path.joinpath(params_file)
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
        params = ConfigBox(params)
    return params