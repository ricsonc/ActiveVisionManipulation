import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('Mask_RCNN')

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
import visualize

import mujoco_datasetp
from toy_train_mujoco import *

if __name__ == '__main__':

    config = MujocoConfig()
    fn = 'dataset.pickle'
    dataset_train = mujoco_dataset.load_dataset(fn)
    dataset_val = mujoco_dataset.load_dataset(fn)
    
    
    model = modellib.MaskRCNN(mode="inference", 
                              config=config,
                              model_dir=MODEL_DIR)

    model_path = model.find_last()[1]

    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, config, 
                               image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    mujoco_datset.make_save('gt.png')
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                dataset_train.class_names, figsize=(8, 8))

    results = model.detect([original_image], verbose=1)

    mujoco_dataset.make_save('pred.png')
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset_val.class_names, r['scores'])    
