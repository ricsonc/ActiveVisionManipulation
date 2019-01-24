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

import large_dataset
from train_mujoco import *

class InferenceConfig(MujocoConfig):
    IMAGES_PER_GPU = 1

if __name__ == '__main__':
    dataset_val = large_dataset.get_val_set()

    config = InferenceConfig()
    
    model = modellib.MaskRCNN(mode="inference", 
                              config=config, 
                              model_dir=MODEL_DIR)

    model_path = model.find_last()[1]

    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)


    # log("original_image", original_image)
    # log("image_meta", image_meta)
    # log("gt_class_id", gt_class_id)
    # log("gt_bbox", gt_bbox)
    # log("gt_mask", gt_mask)

    for i in range(20):
        print("processing image %d" %i)
        image_id = random.choice(dataset_val.image_ids)
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, config, 
                                   image_id, use_mini_mask=False)
    
        large_dataset.make_save('vis/gt_%d.png' % i)
        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                    dataset_val.class_names, figsize=(8, 8))

        results = model.detect([original_image], verbose=1)

        large_dataset.make_save('vis/pred_%d.png' % i)
        r = results[0]

        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                    dataset_val.class_names, r['scores'], figsize=(8,8)) 

        #debug printing
        
        rois = r['rois']
        scores = r['scores']
        print (rois, scores)

        print('='*10)
