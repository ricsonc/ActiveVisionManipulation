
import sys
# /home/ricsonc/hindsight_experience_replay/HER/rcnn/load_rcnn.py 
# -> /home/ricsonc/hindsight_experience_replay/HER/rcnn/Mask_RCNN
sys.path.append(__file__.replace('load_rcnn.py', 'Mask_RCNN'))
sys.path.append(__file__.replace('load_rcnn.py', ''))

import os
import sys
import random
import math
import re
import time
import numpy as np
from config import Config
import utils
import model as modellib
from model import log

from train_mujoco import *


class InferenceConfig(MujocoConfig):
    IMAGES_PER_GPU = 1

def load_rcnn():
    print('LOADING RCNN')
    config = InferenceConfig()

    MODEL_DIR = __file__.replace('load_rcnn.py', 'logs')
    
    model = modellib.MaskRCNN(mode="inference", 
                              config=config, 
                              model_dir=MODEL_DIR)

    model_path = model.find_last()[1]

    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    return model
