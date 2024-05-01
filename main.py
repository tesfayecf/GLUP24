from dotenv import load_dotenv
load_dotenv()

import sys
import tensorflow as tf

from test import test
from train import train
from evaluate import evaluate
from Misc.utils import load_parameters

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    print(e)

def train_model(parameters = None):
    if parameters is None:
        parameters = load_parameters('training_parameters.json')
    
    metric = train(**parameters)    
    return metric

def test_model():
    run_id = "e311b8127ab241338365da3eddc13f59"
    test(run_id)

def evaluate_model():
    run_id = "29ac8739d35e4d7680b4e73080e418f0"
    evaluate(run_id)

if __name__ == "__main__":
    # Execute funtion depending of args
    if len(sys.argv) == 2:
        if sys.argv[1] == "train":
            train_model()
        elif sys.argv[1] == "test":
            test_model()
        elif sys.argv[1] == "evaluate":
            evaluate_model()