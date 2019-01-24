from HER.envs import oc_pusher
import numpy as np 
import math
import random
from scipy.misc import imsave
from ipdb import set_trace as st
        
class BaxterEnv(oc_pusher.BaxterEnv):

    def __init__(self, bbox_noise = 0.0, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.debug = False
        self.bbox_noise = bbox_noise

        self.last_xy = None
        
        self.foo = lambda: None
        self.foo.notfound = 0
        self.foo.found = 0
        self.foo.count = 0.0
        self.foo.err = 0.0
        self.foo.bias = 0.0

    def reset_model(self, *args, **kwargs):
        self.last_xy = None
        rval = super().reset_model(*args, **kwargs)
        self.renderer.reset(reset_env = False) #not sure if this helps
        return rval
        
    def _get_obj_for_obs(self):

        #consider the scenario where bbox is None (aka failed detection)
        bbox = self._get_obj_bbox()
        bbox = self._add_bbox_noise(bbox)
        xy = self.get_xy_from_bbox(bbox)
        return xy

    def _add_bbox_noise(self, box):
        return box
        #noise = self.np_random.normal(loc = 0.0, scale = self.bbox_noise, size = (4,))
        #return box + noise
    
    def _get_obj_bbox(self):
        #if we don't do a throwaway render, sometimes the render_box function
        #returns a garbage output for unknown reasons
        self.renderer.render_rgb()
        mask = self.renderer.render_box()
        return self._get_bbox_from_mask(mask)

    def _get_bbox_from_mask(self, mask, unocc = 1):
        if mask is None:
            return None

        ys, xs = np.where(mask)
        if len(ys) == 0:
            self.foo.notfound += 1
            if random.random() < 0.01 and self.debug:
                print('nf:n ratio is', self.foo.notfound, self.foo.found)
                print('joint is', self._get_joint('cammount', True))
            bbox = 0.0, self.img_params.full_imgW, 0.0, self.img_params.full_imgH
        else:
            self.foo.found += 1
            bbox = np.min(xs), np.max(xs), np.min(ys), np.max(ys)

        #self.last_box = bbox
            
        return np.array(bbox) + self.np_random.normal(loc = 0.0, scale = self.bbox_noise, size = (4,)) * (1-unocc)

    def get_xy_from_bbox(self, box):
        #consider the possibility of box being None (aka failed detection)

        if box is None:
            return self.last_xy 
        
        centerx = (box[0]+box[1])/2
        centery = (box[2]+box[3])/2

        height = self.img_params.full_imgH
        width = self.img_params.full_imgW
        camera_name = self.img_params.cam

        dx = centerx - width/2.0
        dy = centery - height/2.0

        data = self.data
        model = self.sim.model
    
        camera_id = model.camera_name2id(camera_name)
        camera_fovy = model.cam_fovy[camera_id] / 180.0 * np.pi #radians
        focal = height/2.0 / math.tan(camera_fovy/2.0) #units: pixels

        camera_matrix = data.get_camera_xmat(camera_name)
        camera_position = data.get_camera_xpos(camera_name)

        vector_ = np.array([dx, dy, -focal])
        #in mujoco, camera points towards -z
        vector = np.matmul(camera_matrix, vector_)

        dz = vector[-1]
        camera_z = camera_position[-1]
        delta_z = 0.01
        scale = (delta_z-camera_z)/dz

        obj_name = 'box'
        obj_position = data.get_geom_xpos(obj_name)

        pred_xyz = camera_position + vector*scale

        diff = np.linalg.norm(pred_xyz - obj_position)
        bias = pred_xyz-obj_position

        self.foo.count += 1
        self.foo.err += diff
        self.foo.bias += bias
        if self.debug:
            print('avg acc is', self.foo.err/self.foo.count)
            print('avg bias is', self.foo.bias/self.foo.count)
        # ~0.02 for static cam
        # ALSO 0.02 for dynamic cam
        
        #if self.debug or diff > 0.1:
        if self.debug:
            print("pred minus ground truth", pred_xyz - obj_position)
            print('error: ', diff)
            print('box:', box)

            filename = '/tmp/debug/%d.png' % random.randint(1000, 2000)
            img = self.renderer.render_rgb()
            box = np.minimum(box, 63)
            box = box.astype(np.int32)
            img[box[2], box[0]:box[1], :] = 0
            img[box[3], box[0]:box[1], :] = 0
            img[box[2]:box[3], box[0], :] = 0
            img[box[2]:box[3], box[1], :] = 0
            
            imsave(filename, img)

        xy = pred_xyz[:2]

        self.last_xy = xy

        return xy
