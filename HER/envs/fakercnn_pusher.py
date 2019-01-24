from HER.envs import bb_pusher
import numpy as np 
from HER.rcnn import renderer
import random
from keras import backend as K
from HER.rcnn import load_rcnn
import tensorflow as tf
from HER.envs.pusher import _tuple
from ipdb import set_trace as st

class BaxterEnv(bb_pusher.BaxterEnv):

    def __init__(self, *args, aux_rwd = False, test = False, fake_scheme = 'default', **kwargs):
        self.aux_rwd = aux_rwd
        self.fake_scheme = fake_scheme
        self.test = test
        super().__init__(*args, **kwargs)
        if test:
            self.initialize_rcnn()
    
    def reset_model(self):
        #we can't use last_mask or last_box BECAUSE it's misleading once the camera starts moving
        self.last_box_vis = None
        
        # if hasattr(self, 'last_mask'):
        #     del self.last_mask #in case we fail at detection
        # if self.test:
        #     self.last_box = None
        return super().reset_model()

    def calc_reward(self, state, return_success = False):
        rwd, succ = super().calc_reward(state, True)
        if self.aux_rwd and not self.test:
            rwd += state.aux

        if return_success:
            return rwd, succ
        else:
            return rwd

    def _get_obs(self):
        obs = super()._get_obs()
        if self.aux_rwd and not self.test:
            obs.aux = self.last_success
        return obs

    #we need to dig up some old code here...
    def initialize_rcnn(self):
        #need to mess around with a different grpah/session because the rcnn
        #and the HER code runs on separate graphs/sessions
        self.rcnn_graph = tf.Graph()
        self.rcnn_session = tf.Session(graph = self.rcnn_graph)
        K.set_session(self.rcnn_session)
        with self.rcnn_session.as_default():
            with self.rcnn_graph.as_default():
                self.rcnn = load_rcnn.load_rcnn()
    
    def _get_obj_bbox_from_rcnn(self):
        self.renderer.render_rgb() #throwaway
        rgb = self.renderer.render_rgb()

        #####
        K.set_session(self.rcnn_session)
        with self.rcnn_session.as_default():
            with self.rcnn_graph.as_default():
                rcnn_out = self.rcnn.detect([rgb], verbose = 0)[0]
            
        rois = rcnn_out['rois']
        #from ipdb import set_trace as st
        #st()
        
        DEBUG_RCNN = True
        
        if len(rois) == 0 or rcnn_out['scores'][0] < 0.98:
            if DEBUG_RCNN:
                print("rcnn detected nothing")
            box = None
        else:
            #rois are sorted by score, so we take the most confident detection
            y1, x1, y2, x2 = rois[0]
            box = (x1, x2, y1, y2)

            if DEBUG_RCNN:
                print("rcnn detection at", box)
            # self.last_box = box
            self.last_box_vis = box
            
        # return self.last_box
        return box
    
    def _get_obj_bbox(self):
        #if self.test and self.num_step > 1:
        if self.test:
            return self._get_obj_bbox_from_rcnn()
        
        self.renderer.render_rgb() #throwaway
        mask = self.renderer.render_box()
        modal_mask = self.renderer.render_box(override_amodal = False)
        unocc = np.sum(modal_mask) / (np.sum(mask) + 1E-9)

        #complete unocc = 100% success. 75% occluded means 50% detection chance

        if self.num_step == 1: #let's say we alwyas see it on ep 0
            chance_success = 1.0
        else:
            #actually this seems a bit too strict
            if self.fake_scheme == 'default':
                chance_success = np.sqrt(unocc)

            elif self.fake_scheme == 'piecewise':
                if unocc >= 0.2:
                    chance_success = unocc*0.1+0.9 #so 0.92 at 0.2
                else:
                    chance_success = unocc*0.92*5.0
            else:
                raise Exception('unknown fake_scheme')

        if not hasattr(self, 'bar'):
            self.bar = lambda: None
            self.bar.count = 0
            self.bar.succ = 0
            self.bar.unocc = 0.0

        self.bar.count += 1
        self.bar.unocc += unocc

        self.last_success = 0
        if random.random() < chance_success: #if we succeeded
            #self.last_mask = mask
            self.last_success = 0.25
            self.bar.succ += 1
        else: #if we failed
            return None
            
        #elif not hasattr(self, 'last_mask'): #if we failed but don't have last mask
            #self.last_mask = np.zeros((self.img_params.full_imgH, self.img_params.full_imgW))


        if self.debug:
            print('detect rate', self.bar.succ/self.bar.count)
            print('unocc', self.bar.unocc/self.bar.count)
        
        #return self._get_bbox_from_mask(self.last_mask, unocc)
        return self._get_bbox_from_mask(mask, unocc)
