from utils import load_params_util
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

import argparse


from utils.load_params import load_params

## Stub for unit tests

dataset_url = params.data_collection.dataset_url
data_dir = Path(params.data_collection.data_dir)
data_dir.mkdir(exist_ok=True)
orig_dirname = params.data_collection.orig_dirname
new_dirname = params.data_collection.new_dirname


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params_util(params_path)
    data_load(params)
