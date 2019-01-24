import numpy as np
from scipy.misc import imsave
from functools import reduce

class Renderer(object):
    def __init__(self, env):
        self.env = env
        self.data = env.sim.data
        self.model = env.sim.model

        self.is_amodal = self.env.img_params.is_amodal
        
        self.cam = self.env.img_params.cam
        self.width = self.env.img_params.full_imgW
        self.height = self.env.img_params.full_imgH

        # by default, sites are set to be fully transparent
        # this option allows for sites to act as occluders
        self.visible_sites = ['distractor', 'distractor1', 'distractor2', 'distractor3']

    def set_lighting(self):
        self.tmp_diffuse = self.model.light_diffuse[0].copy()
        self.tmp_ambient = self.model.light_ambient[0].copy()
        self.tmp_specular = self.model.light_specular[0].copy()
        
        assert len(self.model.light_diffuse) == 1
        self.model.light_diffuse[0] = np.zeros_like(self.model.light_diffuse[0])
        self.model.light_ambient[0] = np.ones_like(self.model.light_ambient[0])
        self.model.light_specular[0] = np.zeros_like(self.model.light_specular[0])

    def unset_lighting(self):
        self.model.light_diffuse[0] = self.tmp_diffuse
        self.model.light_ambient[0] = self.tmp_ambient
        self.model.light_specular[0] = self.tmp_specular

    def set_color(self, obj_name, override_amodal = None):
        self.geom_rgba = self.model.geom_rgba.copy()
        self.site_rgba = self.model.site_rgba.copy()
        
        obj_id = self.model.geom_name2id(obj_name)

        amodal = self.is_amodal
        if override_amodal is not None:
            amodal = override_amodal
        
        for i in range(len(self.model.geom_rgba)):
            alpha = 0.0 if amodal else 1.0
            self.model.geom_rgba[i] = np.array([0.0, 0.0, 0.0, alpha])

        for i in range(len(self.model.site_rgba)):
            site_name = self.model.site_names[i]
            alpha = 0.0
            if site_name in self.visible_sites and not amodal:
                alpha = 1.0
            self.model.site_rgba[i] = np.array([0.0, 0.0, 0.0, alpha])

        self.model.geom_rgba[obj_id] = np.array([1.0, 1.0, 1.0, 1.0])

    def unset_color(self):
        for i in range(len(self.model.geom_rgba)):
            self.model.geom_rgba[i] = self.geom_rgba[i]
        for i in range(len(self.model.site_rgba)):
            self.model.site_rgba[i] = self.site_rgba[i]

    def set_color_for_rgb(self):
        # this mainly involves setting whether the sites should be visible 
        for i in range(len(self.model.site_rgba)):
            site_name = self.model.site_names[i]
            
            if site_name in self.visible_sites:
                alpha = 1.0
            else:
                alpha = 0.0
            self.model.site_rgba[i][3] = alpha

    def unset_color_for_rgb(self):
        for i in range(len(self.model.site_rgba)):
            self.model.site_rgba[i][3] = 1.0
        
    def render_obj_(self):
        img = self.env.render(
            mode='rgb_array',
            depth=False,
            width=self.width,
            height=self.height,
            camera_name=self.cam
        )
        return img

    def render_obj(self, obj_name, override_amodal = None):
        self.set_lighting()
        self.set_color(obj_name, override_amodal)
        result = self.render_obj_()
        self.unset_lighting()
        self.unset_color()

        return self.postprocess(result)

    def postprocess(self, result):
        result = np.mean(result, axis = 2)
        result = (result > 200).astype(np.float32)
        return result
    
    def render_box(self, **kwargs):
        return self.render_obj('box', **kwargs)

    def render_gripper(self, **kwargs):
        f = lambda x: self.render_obj(x, **kwargs)
        return reduce(np.maximum, map(f, ['grip1', 'grip2', 'grip3', 'grip4']))

    def render_both(self):
        box = self.render_box()
        gripper = self.render_gripper()        
        zero = np.zeros_like(box)
        return np.stack([box, zero, gripper], axis = 2)

    def render_rgb(self):
        self.set_color_for_rgb()
        img = self.env.render(
            mode='rgb_array',
            depth=False,
            width=self.width,
            height=self.height,
            camera_name=self.cam
        )
        self.unset_color_for_rgb()
        return img

    def reset(self, reset_env = True):
        if reset_env:
            self.env.reset()
        self.data = self.env.sim.data
        self.model = self.env.sim.model

    def rand_state(self):
        #this is necessary for active camera envs...
        self.env.rand_state()

if __name__ == '__main__':
    from HER.envs import baxter_orient_left_cts_img, baxter_orient_left_cts_shapenet
    #env = baxter_orient_left_cts_base.BaxterEnv()
    env = baxter_orient_left_cts_shapenet.BaxterEnv()
    renderer = Renderer(env)

    renderer.reset()

    box = renderer.render_box()
    imsave('box.png', box)    
    exit()
    
    img = renderer.render_both()
    imsave('test.png', img)

    img = renderer.render_rgb()
    imsave('rgb.png', img)
