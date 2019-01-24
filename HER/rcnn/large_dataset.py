import sys

if ('HER/examples/run.py' not in sys.argv[0]) and ('HER/examples/visualize.py' not in sys.argv[0]):
    try:
        import matplotlib    
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import visualize        
    except:
        print ('import failed')
        
from Mask_RCNN.utils import Dataset
import numpy as np

import sys
sys.path.append('Mask_RCNN')

from time import time
import pickle
import os

from scipy.misc import imread, imsave

#things to check
#1. need to preprocess/normalize image/mask input?

NUMTHREADS=8
DATADIR = 'ap-v3_data'

#a hack to get around not being able to use tkinter    
def make_save(name):
    plt.show = lambda: (plt.savefig(name, bbox_inches = 'tight', dpi = 300), plt.clf())
    
class Paths(object):
    def __init__(self, datadir):
        self.datadir = datadir
        if not os.path.exists(datadir):
            os.mkdir(datadir)
            
        self.maskdir = os.path.join(datadir, 'mask')
        if not os.path.exists(self.maskdir):
            os.mkdir(self.maskdir)

        self.imgdir = os.path.join(datadir, 'img')
        if not os.path.exists(self.imgdir):
            os.mkdir(self.imgdir)

    def mask_fn_for_id(self, i):
        return os.path.join(self.maskdir, '%d.npz' % i)

    def img_fn_for_id(self, i):
        return os.path.join(self.imgdir, '%d.png' % i)

    def save_img_to_id(self, i, img):
        pth = self.img_fn_for_id(i)
        imsave(pth, img)

    def load_img_from_id(self, i):
        pth = self.img_fn_for_id(i)
        return imread(pth)

    def save_mask_to_id(self, i, mask):
        pth = self.mask_fn_for_id(i)
        np.savez_compressed(pth, mask)
        
    def load_mask_from_id(self, i):
        pth = self.mask_fn_for_id(i)
        return np.load(pth)['arr_0']

class MujocoData(Dataset, Paths):

    def __init__(self, datadir='data'):
        Dataset.__init__(self)
        Paths.__init__(self, datadir)
        
    def load_mujoco(self, start_id = 0, end_id = 1000):
        self.add_class("mujoco", 1, "box")

        for i in range(start_id, end_id):
            self.add_image("mujoco", i, None)
    
    def load_image(self, i):
        return self.load_img_from_id(i)

    def load_mask(self, i):
        mask = self.load_mask_from_id(i)
        num_objs = mask.shape[-1]
        cls = cls = np.array([1]*num_objs).astype(np.int32)
        return mask, cls

class GenerateData(Paths):
    def __init__(self, renderer, datadir):
        self.renderer = renderer
        super().__init__(datadir)

    def generate(self, count=1000, tid = None):
        print ('tid is', tid)
        iterator = (range(count)
                    if tid is None
                    else range(tid, count, NUMTHREADS))
        
        for i in iterator:
            print("generating image %d" %i)
            img, masks = self._generate()
            self.save_mask_to_id(i, masks)
            self.save_img_to_id(i, img)

    def _generate(self):
        self.renderer.reset()
        self.renderer.rand_state()

        img = self.renderer.render_rgb()
        box = self.renderer.render_box()
        box_modal = self.renderer.render_box(override_amodal = False)

        THRESHOLD = 50 #number of pixels visible
        exists_box = np.sum(box) > THRESHOLD

        #modality
        RATIO_THRESHOLD = 0.1
        exists_box = exists_box and (float(np.sum(box_modal)) / np.sum(box) > RATIO_THRESHOLD)

        cls = []
        masks = []
        if exists_box:
            masks.append(box)
            cls.append(1)
        else:
            print('warning: no box found')

        if exists_box:
            masks = np.stack(masks, axis = 2)
        else:
            h, w, _ = img.shape
            masks = np.zeros((h, w, 0)).astype(np.float32)

        #cls = np.array(cls).astype(np.int32)

        return img, masks#, cls

def visualize_data(data):
    image_ids = data.image_ids[:10]
    for image_id in image_ids:
        image = data.load_image(image_id)
        mask, class_ids = data.load_mask(image_id)
        #print(class_ids)

        make_save("debug/%s" % image_id)
        visualize.display_top_masks(image, mask, class_ids, data.class_names, limit = 1)

def get_train_set():
    datadir=DATADIR
    data = MujocoData(datadir)
    data.load_mujoco(0, 8000)
    data.prepare()
    return data
    
def get_val_set():
    datadir=DATADIR
    data = MujocoData(datadir)
    data.load_mujoco(8000, 10000)
    #data.load_mujoco(0, 50)
    data.prepare()
    return data

def generate_wrapper(tid = None):
    import HER.envs
    import gym
    env = gym.make('active_pusher-v3')
        
    from renderer import Renderer
    renderer = Renderer(env)
    generator = GenerateData(renderer, datadir)
    generator.generate(count, tid)

if __name__ == '__main__':    

    do_generate = True
    do_vis = False
    count = 10000
    
    datadir=DATADIR
    
    if do_generate:
        if False:
            generate_wrapper()
        else:
            from multiprocessing import Process
            processes = []
            for j in range(NUMTHREADS):
                process = Process(target = lambda: generate_wrapper(j))
                process.start()
                processes.append(process)
            for process in processes:
                process.join()
            
    if do_vis:
        data = MujocoData(datadir)
        data.load_mujoco(0, count)
        data.prepare()

        visualize_data(data)
