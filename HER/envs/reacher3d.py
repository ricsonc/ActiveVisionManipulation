import numpy as np
from HER.envs import reacher2d
from gym.envs.robotics.utils import mocap_set_action

class BaxterEnv(reacher2d.BaxterEnv):
    """cts env, 6dim
    state space: absolute state space position of gripper, target
    random restarts for target on the table
    reward function: - 1(not reaching)
    actions: (delta_x, delta_y) 5cm push
    max_num_steps = 20
    """
    def __init__(self, max_len=50, obs_dim=6, action_dim=3, filename="mjc/reacher.xml", space_dim=3):
        super(BaxterEnv, self).__init__(max_len=max_len, obs_dim=obs_dim, action_dim=action_dim, filename=filename, space_dim=space_dim)

      

    ## gym methods

    def reset_model(self,
        # initial config of the end-effector
            gripper_pos = np.array([0.6 , 0.3 , 0.15]),
            ctrl = None,
            add_noise = True
        ):
        
        return super(BaxterEnv, self).reset_model(gripper_pos = gripper_pos, ctrl = ctrl, add_noise = add_noise)
    

    ## my methods
    def step(self, action):
        
        self.num_step += 1
        
        ## parsing of primitive actions
        delta_x, delta_y, delta_z = action
        
        # cap deltas to be between -1, 1
        delta_x = max(-1, min(1, delta_x))
        delta_y = max(-1, min(1, delta_y))
        delta_z = max(-1, min(1, delta_z))

        x, y, z = self.get_mocap_state()
        x += delta_x*0.05
        y += delta_y*0.05
        z += delta_z*0.05
        
        out_of_bound = (x<0.3 or x>0.8) or (y<0.0 or y>0.6) or (z<0.1 or z>0.25)
        
        if not out_of_bound:
            delta_pos = np.array([delta_x*0.05 , delta_y*0.05 , delta_z*0.05])
            delta_quat = np.array([0.0, 0.0 , 1.0, 0.])
            delta = np.concatenate((delta_pos, delta_quat))
            mocap_set_action(self.sim, delta)
            self.do_simulation()
        

        ob = self._get_obs()
        total_reward = self.calc_reward(ob)
        
        ## getting state
        info = {"done":None}
        if total_reward == 0:
            done = True
            info["done"] = "goal reached"
        elif (self.num_step > self.max_num_steps):
            done = True
            info["done"] = "max_steps_reached"
        else: 
            done = False

        info['absolute_ob'] = ob.copy()
        
        return ob, total_reward, done, info
                                        