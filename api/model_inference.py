
import os
import sys
import tensorflow as tf
from pathlib import Path
src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

from utils.custom_metrics import StatefullMultiClassFBeta
from utils.load_params import load_params


dependencies = {
    'StatefullMultiClassFBeta': StatefullMultiClassFBeta
}

params = load_params('params.yaml')
best_model = params.model_inference.best_model_name
model_location = params.model_inference.saved_model_location

model = tf.keras.models.load_model(model_location + "/" + best_model, custom_objects=dependencies)

