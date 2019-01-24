from HER.envs import fakercnn_pusher
import numpy as np 
import quaternion
from HER.envs.pusher import _tuple
import random
from ipdb import set_trace as st

class BaxterEnv(fakercnn_pusher.BaxterEnv):

    def __init__(self, *args, rot = False, rot_scale = 0.2, pos_scale = 0.03, bound = 0.2,
                 img = True, randcam = False,
                 **kwargs):

        obs_dim = 10 if rot else 6
        if 'not_oc' in kwargs and kwargs['not_oc']:
            obs_dim += 2
        action_dim = 8 if rot else 4
        self.cam_rot = rot
        self.rot_scale = rot_scale
        self.pos_scale = pos_scale
        self.bound = bound
        self.randcam = randcam
        super().__init__(*args, obs_dim = obs_dim, action_dim = action_dim, img = img, **kwargs)

    #pusher.BaxterEnv.reset_model already resets camera pose, so we're in the clear there.
        
    def step(self, action):
        gripper_action = action[:2]
        camera_action = action[2:]
        self.apply_camera_delta(camera_action)
        return super().step(gripper_action)

    def _get_obs(self):
        partial_state = super()._get_obs()
        camera_state = self.get_camera_state()
        return self.combine_camera_state(partial_state, camera_state)
    
    def combine_camera_state(self, partial_state, camera_state):
        assert isinstance(partial_state, tuple)
        #ignore the 0th elemtn in the camera_state (the full thing)

        coordinates = [partial_state[0], camera_state[2]]
        if self.cam_rot:
            #quaternion
            coordinates.append(camera_state[1])
        
        rval = _tuple(
            (np.concatenate(coordinates),
             partial_state[1]),
            info = partial_state.info,
            aux = partial_state.aux
        )
        return rval
    
    def see(self):
        import matplotlib.pyplot as plt
        rgb = self.renderer.render_rgb()
        plt.imshow(rgb)
        plt.show()

    #input: np array and scalar
    def raw_rot_quat(self, axis, magnitude):
        axis = axis/(np.linalg.norm(axis)+1E-9)
        axisquat = np.quaternion(*axis) #4 elems
        return np.exp(0.5 * magnitude * axisquat)
    
    #input: np array, scalar, and np array
    def raw_compose_quat(self, axis, magnitude, original):
        original = np.quaternion(*original)
        return self.raw_rot_quat(axis, magnitude) * original

    #output: a usable np array
    def compose_quat(self, axis, magnitude, original):
        return quaternion.as_float_array(self.raw_compose_quat(axis, magnitude, original))

    def valid_camera_pos(self, xy):
        original_pos = np.array([0.2, 0.0])
        distance_from_original = np.linalg.norm(xy - original_pos)

        #camera bounds are 20cm in each direction
        if distance_from_original > self.bound: 
            return False

        center = ((self.target_range_min + self.target_range_max)/2)[:2]
        original_dist_from_center = np.linalg.norm(original_pos - center)
        new_dist_from_center = np.linalg.norm(xy-center)
        diff = original_dist_from_center - new_dist_from_center

        #camera cannot get too close to the env
        if diff > 0.1:
            return False

        return True

    def get_camera_state(self):
        state = self._get_joint('cammount', True)
        rot = state[3:] #quaternion
        pos = state[:2] #no vertical component
        return state, rot, pos
    
    def apply_camera_delta(self, action):

        if self.cam_rot:
            rot_delta_axis = action[:3]
            rot_delta_mag = action[3] * self.rot_scale
            pos_delta = action[4:] * self.pos_scale #max 3 cm per step
        else:
            rot_delta_axis = np.zeros(3)
            rot_delta_mag = 0.0
            pos_delta = action * self.pos_scale

        if self.randcam:
            pos_delta = np.random.uniform(size = 2, low = -1, high = +1) * self.pos_scale

        state, rot_original, pos_original = self.get_camera_state()

        new_pos = pos_original + pos_delta
        if not self.valid_camera_pos(new_pos):
            new_pos = pos_original

        if self.cam_rot:
            new_quat = self.compose_quat(rot_delta_axis, rot_delta_mag, rot_original)
            state[3:] = new_quat
        
        state[:2] = new_pos

        self._set_joint('cammount', state)

    def rand_state(self):
        #is this good enough?... probably not
        steps = random.randint(0, 20)
        action = np.random.uniform(2)*2-1
        for _ in range(steps):
            if random.random() > 0.9:
                #change action w/ 10% chance
                action = np.random.uniform(2)*2-1
            self.apply_camera_delta(action)
            self.sim.step()
