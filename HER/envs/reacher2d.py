import numpy as np
import os 
import os.path as osp
import signal 

from gym import utils
from gym import spaces
from gym.envs.robotics.utils import mocap_set_action

from HER.envs import mujoco_env

class BaxterEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """cts env, 4dim
    state space: absolute state space position of gripper, target
    random restarts for target on the table
    reward function: - 1(not reaching)
    actions: (delta_x, delta_y) 5cm push
    starting state: (0.63, 0.2, 0.59, 0.27, 0.55, 0.3)
    max_num_steps = 50
    """
    def __init__(self, max_len=50, obs_dim=4, action_dim=2, filename="mjc/reacher.xml", space_dim=2):
        dirname = os.path.dirname(os.path.abspath(__file__)) 
        
        ## mujoco things
        # task space action space
        self.action_space = spaces.Box(-1., 1., shape=(action_dim,), dtype='float32')

        # task space observation space
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype='float32')

        # motion dim
        self.space_dim = space_dim

        # load the model from xml
        mujoco_env.MujocoEnv.__init__(self, os.path.join(dirname, filename))
        utils.EzPickle.__init__(self)

        # env specific params
        self.max_num_steps = max_len
        print("INIT DONE!")
      
    def copy_obs(self, x):
        return x.copy()
        
    ## gym methods

    def reset_model(self,
        # initial config of the end-effector
            gripper_pos = np.array([0.6 , 0.3 , 0.15]),
            gripper_quat = np.array([0.0, 0.0 , 1.0, 0.]),
            ctrl = None,
            add_noise = True
        ):
        
        if add_noise:
            target_qpos = self.sim.data.get_joint_qpos('target') 
            assert target_qpos.shape == (7,)
            target_qpos[:self.space_dim] = self.get_rand()
            # print("Setting target qpos to ", target_qpos)
            self.data.set_joint_qpos('target', target_qpos)
            gripper_pos[:self.space_dim] = self.get_rand()
            
        self.apply_action(gripper_pos, gripper_quat, ctrl)
        
        self.num_step = 1
        
        ob = self._get_obs()
        return ob

    def viewer_setup(self):
        # cam_pos = np.array([0.1, 0.0, 0.7, 0.01, -45., 0.])
        cam_pos = np.array([1.0, 0.0, 0.7, 0.5, -45, 180])
        self.set_cam_position(cam_pos)

    def _get_obs(self):
        ee_pos = self.sim.data.get_site_xpos('grip')[:self.space_dim]
        target_pos = self.sim.data.get_site_xpos('target')[:self.space_dim]
        
        state = np.concatenate([ee_pos, target_pos])
        return state

    ## my methods
    def apply_action(self, pos, quat=np.array([0.0, 0.0 , 1.0, 0.]), ctrl = None):  
        self.sim.data.set_mocap_pos('robot0:mocap', pos)
        self.sim.data.set_mocap_quat('robot0:mocap', quat)
        # large motion take more env steps
        self.do_simulation(ctrl, 1)    

    def close_gripper(self, left_gap=0):
        """
        Implement this in subclass with grasping capabilities.
        """
        raise NotImplementedError

    def step(self, action):
        
        self.num_step += 1
        
        ## parsing of primitive actions
        delta_x, delta_y = action
        
        # cap deltas to be between -1, 1
        delta_x = max(-1, min(1, delta_x))
        delta_y = max(-1, min(1, delta_y))

        x, y = self.get_mocap_state()
        x += delta_x*0.05
        y += delta_y*0.05
        
        out_of_bound = (x<0.3 or x>0.8) or (y<0.0 or y>0.6)
        
        if not out_of_bound:
            delta_pos = np.array([delta_x*0.05 , delta_y*0.05 , 0.0])
            delta_quat = np.array([0.0, 0.0 , 1.0, 0.])
            delta = np.concatenate((delta_pos, delta_quat))
            mocap_set_action(self.sim, delta)
            self.do_simulation()
        
        ob = self._get_obs()

        total_reward, success = self.calc_reward(ob, True)
        
        ## getting state
        info = {"done":None}
        if success:
            done = True
            info["done"] = "goal reached"
        elif (self.num_step > self.max_num_steps):
            done = True
            info["done"] = "max_steps_reached"
        else: 
            done = False

        info['absolute_ob'] = self.copy_obs(ob)
        
        return ob, total_reward, done, info
                                        
    def apply_hindsight(self, states, actions, goal_state):
        '''generates hindsight rollout based on the goal
        '''
        goal = goal_state[:self.space_dim]    ## this is the absolute goal location = gripper last loc
        # enter the last state in the list
        states.append(goal_state)
        num_tuples = len(actions)

        her_states, her_rewards = [], []
        # make corrections in the first state 
        states[0][-self.space_dim:] = goal.copy()
        her_states.append(states[0])
        for i in range(1, num_tuples + 1):
            state = states[i]
            state[-self.space_dim:] = goal.copy()    # copy the new goal into state
            reward = self.calc_reward(state)
            her_states.append(state)
            her_rewards.append(reward)

        return her_states, her_rewards
    
    def calc_reward(self, state):
        # this functions calculates reward on the current state
        gripper_pose = state[:self.space_dim]
        target_pose = state[-self.space_dim:] 
        
        ## reward function definition
        reward_reaching_goal = np.linalg.norm(gripper_pose- target_pose) < 0.02
        total_reward = -1*(not reward_reaching_goal)
        return total_reward
