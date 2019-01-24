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

import mujoco_dataset

class MujocoConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "mujoco"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
      
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

if __name__ == '__main__':

    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    config = MujocoConfig()
    
    #dataset_train = mujoco_dataset.make_dataset(count = 6)
    #dataset_val = mujoco_dataset.make_dataset(count = 2)
    fn = 'dataset.pickle'
    dataset_train = mujoco_dataset.load_dataset(fn)
    dataset_val = mujoco_dataset.load_dataset(fn)
    
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)    


    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)


    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=1, 
                layers='heads')


    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also 
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE / 10,
                epochs=2, 
                layers="all")
