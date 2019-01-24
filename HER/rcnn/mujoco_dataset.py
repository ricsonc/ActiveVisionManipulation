import matplotlib
matplotlib.use('Agg')

from Mask_RCNN.utils import Dataset
from renderer import Renderer

import numpy as np

import matplotlib.pyplot as plt
    
import sys
sys.path.append('Mask_RCNN')
import visualize

from time import time
import pickle
import os

class MujocoData(Dataset):

    def __init__(self, renderer):
        self.renderer = renderer
        self.imgs = {}
        self.masks = {}
        self.cls = {}
        super().__init__()
        
    def load_mujoco(self, count = 1000):
        
        self.add_class("mujoco", 1, "box")
        self.add_class("mujoco", 2, "gripper")

        if count == -1:
            return

        for i in range(count):
            print(i,)
            self.generate_img(i)

    def load_image(self, img_id):
        return self.imgs[img_id]

    def load_mask(self, img_id):
        return self.masks[img_id], self.cls[img_id]    

    def generate_img(self, img_id):
        img, masks, cls = self.generate_img_()
        self.add_image("mujoco", img_id, None)

        self.imgs[img_id] = img
        self.masks[img_id] = masks        
        self.cls[img_id] = cls

    def generate_img_(self):
        self.renderer.reset()
        img = self.renderer.render_rgb()
        box = self.renderer.render_box()
        gripper = self.renderer.render_gripper()
        #the renderer is pretty slow

        THRESHOLD = 50 #number of pixelx
        exists_box = np.sum(box) > THRESHOLD
        exists_gripper = np.sum(gripper) > THRESHOLD

        cls = []
        masks = []
        if exists_box:
            masks.append(box)
            cls.append(1)
        if exists_gripper:
            masks.append(gripper)
            cls.append(2)

        masks = np.stack(masks, axis = 2)
        cls = np.array(cls).astype(np.int32)

        return img, masks, cls

    #untested functions
    def save(self, name):
        data = [self.imgs, self.masks, self.cls]
        with open(name, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, name):
        with open(name, 'rb') as f:
            data = pickle.load(f)

        self.imgs, self.masks, self.cls = data
        for i in range(len(self.imgs)):
            self.add_image("mujoco", i, None)    

#a hack to get around not being able to use tkinter    
def make_save(name):
    plt.show = lambda: plt.savefig(name, bbox_inches = 'tight', dpi = 300)

def make_dataset(count = 1000):
    from HER.envs import baxter_orient_left_cts_base    
    env = baxter_orient_left_cts_base.BaxterEnv()
    renderer = Renderer(env)
    data = MujocoData(renderer)
    data.load_mujoco(count)
    data.prepare()
    return data
    #just makes a basic mujoco dataset 

def load_dataset(name):
    data = MujocoData(None)
    data.load(name)
    data.load_mujoco(-1)
    data.prepare()
    return data
    #just makes a basic mujoco dataset 
    
if __name__ == '__main__':    

    fn = 'dataset.pickle'
    if not os.path.exists(fn):
        data = make_dataset(count = 256)
        data.save(fn)
    else:
        data = load_dataset(fn)
        
    print('loaded dataset')
    
    image_ids = data.image_ids[:10]
    for image_id in image_ids:
        image = data.load_image(image_id)
        mask, class_ids = data.load_mask(image_id)

        make_save("%s" % image_id)
        visualize.display_top_masks(image, mask, class_ids, data.class_names, limit = 2)
