from HER.envs import close_pusher
import numpy as np 
from gym import spaces
from ipdb import set_trace as st
from HER.envs.pusher import _tuple, _ndarray

# in oc pusher, simply storing the ground truth location of the box is no longer
# sufficient to recover the reward, because we have neither the estimated box
# location nor the absolute location of the target. instead, we directly store the
# delta between the box and the gt position. 

# then, when carrying out hindsight, we can use the ground truth delta to reconstruct
# the observed states

from ipdb import set_trace as st

class BaxterEnv(close_pusher.BaxterEnv):
    
    def __init__(self, max_len=20, obs_dim = 4, not_oc = False, no_obj = False, **kwargs):
        self.not_oc = not_oc
        self.no_obj = no_obj
        super(BaxterEnv, self).__init__(max_len=max_len, obs_dim = obs_dim, **kwargs)
        
    def _get_obs(self):
        ee_pos = self._get_ee_xy()
        obj_pos = self._get_obj_for_obs()
        gt_obj_pos = self._get_obj_xy()
        target_pos = self._get_target_xy()

        if 'active_pusher' in str(self.__class__):
            if obj_pos is None:
                obj_pos = ee_pos
        
        state = np.concatenate([ee_pos - obj_pos, target_pos - obj_pos])
        delta = target_pos - gt_obj_pos
        
        if self.not_oc:
            state = np.concatenate([ee_pos, obj_pos, target_pos])
            if self.no_obj:
                state[2:4] = 0.0 #remove obj position from state info
            delta = gt_obj_pos
        
        if self.img:
            return _tuple((state, self.get_rgb()), info = delta)
        else:
            return _ndarray(state, info = delta)

    def obj_target_dist_for_state(self, state):
        if self.not_oc:
            return close_pusher.BaxterEnv.obj_target_dist_for_state(self, state)
        
        return np.linalg.norm(state.info)

    def get_hindsight_state(self, state, goal_state):
        if self.not_oc:
            return close_pusher.BaxterEnv.get_hindsight_state(self, state, goal_state)
            
        delta = goal_state.info

        if self.img:
            state[0][-2:] -= delta
        else:
            state[-2:] -= delta

        #also modify ground truth distance from target!
        state.info -= delta
            
        return state
