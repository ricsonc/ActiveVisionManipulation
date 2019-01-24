from HER.envs import reacher2d
import numpy as np 
from HER.rcnn import renderer
from gym import spaces
import random

# we need the ability to tack on additional info onto our states, which are only
# partial observations of the environment. this is necessary because we need to 
# be able to compute the reward based off of just the state (which is something
# HER needs to be able to replay episodes with a different goal.
# we subclass both tuple and also ndarray in order to achieve this. 

# for the pusher env, the additional information we require is gt_obj_pos, since
# we can then simply compute the distance to the target (which we have gt of). 
# note how `obj_target_dist_for_state` reflects this

class _tuple(tuple):
    def __new__(cls, args, info = None, aux = None):
        return super().__new__(cls, args)

    def __init__(self, args, info = None, aux = None):
        self.info = info
        self.aux = aux

    def copy(self):
        return _tuple([arg.copy() for arg in self], info = self.info.copy(), aux = self.aux)

class _ndarray(np.ndarray):
    def __new__(cls, arr, info = None, aux = None):
        x = np.asarray(arr).view(cls)
        x.info = info
        x.aux = aux
        return x

    def __array_finalize__(self, x):
        if x is None:
            return
        self.info = getattr(x, 'info', None)
        self.aux = getattr(x, 'aux', None)
        
    def copy(self):
        rval =  super().copy()
        rval.info = self.info
        rval.aux = self.aux
        return rval

class ImgParams:
    def __init__(self, imgH = 64, imgW = 64,
                 full_imgH = 64, full_imgW = 64, margin = 2.0,
                 cam = 'cam4', is_amodal = True):
        
        self.imgH = imgH
        self.imgW = imgW
        self.full_imgH = full_imgH
        self.full_imgW = full_imgW
        self.margin = margin
        self.cam = cam
        self.is_amodal = is_amodal

class BaxterEnv(reacher2d.BaxterEnv):
    
    def __init__(self, max_len=20, filename = 'mjc/pusher.xml', obs_dim = 6, action_dim = 2,
                 img = False, img_params = None):

        self.base_name = filename
        super(BaxterEnv, self).__init__(
            max_len=max_len, obs_dim=obs_dim, action_dim=action_dim, filename=filename, space_dim=2
        )


        self.img_params = ImgParams() if img_params is None else img_params
        self.renderer = renderer.Renderer(self)
            
        if img:
            img_space = spaces.Box(
                low = 0.0, high = 1.0,
                shape = (self.img_params.full_imgH, self.img_params.full_imgW, 3),
                dtype = np.float32
            )
            self.observation_space = spaces.Tuple((self.observation_space, img_space))

        self.img = img
        
    def reset_model(self, 
        # define one pose in hand 
        gripper_pos = np.array([0.6 , 0.3 , 0.12]),
        add_noise = False,
        randomize_obj = True):

        #we need to reset the camera mount otherwise the camera mount slowly drifts downwards
        initial_camera_pos = np.array([0.2, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])
        self._set_joint('cammount', initial_camera_pos)

        if randomize_obj:
            # randomize obj loc
            object_qpos = self.sim.data.get_joint_qpos('box') 
            assert object_qpos.shape == (7,)
            object_qpos[:self.space_dim] = self.get_rand()
            # fill obj to fully on ground
            object_qpos[3:] = np.array([1., 0.,0.,0.])
            self.data.set_joint_qpos('box', object_qpos)

            while 1:
                target_qpos = self.sim.data.get_joint_qpos('target') 
                target_qpos[:self.space_dim] = self.get_rand()
                if np.linalg.norm(target_qpos[:self.space_dim] - object_qpos[:self.space_dim]) < 0.05:
                    self.data.set_joint_qpos('target', target_qpos)
                    break
                
            gripper_pos[:self.space_dim] = self.get_rand()
            while(np.linalg.norm(gripper_pos[:self.space_dim] - object_qpos[:self.space_dim]) < 0.1):
                gripper_pos[:self.space_dim] = self.get_rand()

        if 'distractor' in self.base_name:
            objxy = self._get_obj_xy()
            tgtxy = self._get_target_xy()
            x = min(objxy[0], tgtxy[0])-0.1
            y = (objxy[1] + tgtxy[1])/2.0 - 0.1
            xy = np.array([x,y])

            behavior = 'new'
            if behavior == 'old':
                xy += self.np_random.uniform(-0.05, 0.05, size = 2)
                dist_qpos = self.sim.data.get_joint_qpos('distractor1')
                dist_qpos[:self.space_dim] = xy
                dist_qpos[self.space_dim] = 0.0
                dist_qpos[3:] = np.array([1., 0.,0.,0.])
                self.data.set_joint_qpos('distractor1', dist_qpos)

            elif behavior == 'new':
                for i in range(1,4):
                    d_name = 'distractor%d' % i
                    
                    if random.random() > 0.5:
                        #doesn't help for some reason
                        self.data.set_joint_qpos(d_name, np.zeros_like(self.data.get_joint_qpos(d_name)))
                        continue

                    dxy = self.np_random.uniform(-0.08, 0.08, size = 2)
                    dxy[0] /= 2.
                    xy_ = xy + dxy
                                       
                    dist_qpos = self.sim.data.get_joint_qpos(d_name)
                    dist_qpos[:self.space_dim] = xy_
                    dist_qpos[self.space_dim] = 0.0
                    dist_qpos[3:] = np.array([1., 0.,0.,0.])

                    self.data.set_joint_qpos(d_name, dist_qpos)

                    #set random color:
                    d_id = self.sim.model.site_name2id(d_name)
                    new_color = np.array([random.random()/2.0, random.random(), random.random(), 1.0])
                    self.sim.model.site_rgba[d_id] = new_color

            else:
                print('bad!')
            
        return super(BaxterEnv, self).reset_model(gripper_pos = gripper_pos,
            add_noise = add_noise)

    def _get_obs(self, noimg = False):
        ee_pos = self._get_ee_xy()
        obj_pos = self._get_obj_for_obs()
        gt_obj_pos = self._get_obj_xy()
        target_pos = self._get_target_xy()
        
        state = np.concatenate([ee_pos, obj_pos, target_pos])

        if self.img:
            return _tuple((state, self.get_rgb()), info = gt_obj_pos)
        else:
            return _ndarray(state, info = gt_obj_pos)

    def close_gripper(self, gap=0):
        self.data.ctrl[0] = (gap+1)*0.04
        self.data.ctrl[1] = -(gap+1)*0.04
        
    def step(self, action):
        
        ob, total_reward, done, info = super(BaxterEnv, self).step(action)
        
        x,y = self._get_obj_xy()
        if (x<0.3 or x>0.8) or (y<0.0 or y>0.6):
            done = True
            info['done'] = 'unstable simulation'
            total_reward -= (self.max_num_steps - self.num_step) + 2

        return ob, total_reward, done, info

    def apply_hindsight(self, states, actions, goal_state):
        '''generates hindsight rollout based on the goal
        '''
        states.append(goal_state)
        num_tuples = len(actions)

        her_states, her_rewards = [], []

        for i in range(num_tuples+1):
            her_state = self.get_hindsight_state(
                self.copy_obs(states[i]),
                self.copy_obs(goal_state)
            )
            her_states.append(her_state)
            if i > 0:
                her_rewards.append(self.calc_reward(her_state))

        return her_states, her_rewards
        
    def get_hindsight_state(self, state, goal_state):
        if self.img:
            goal_state = goal_state[0]
        goal = goal_state[self.space_dim:2*self.space_dim]
        if False:
            if self.img:
                state[0][-self.space_dim:] = goal
            else:
                state[-self.space_dim:] = goal
        else: #camera
            if self.img:
                state[0][-4:-2] = goal
            else:
                state[-4:-2] = goal
        return state

    def calc_reward(self, state, return_success = False): #uh oh... state here probably has an error to it
        ## reward function definition
        reward_reaching_goal = self.obj_target_dist_for_state(state) < 0.02
        total_reward = -1*(not reward_reaching_goal)

        if return_success:
            return total_reward, reward_reaching_goal
        else:
            return total_reward

    def obj_target_dist(self):
        return np.linalg.norm(self._get_obj_xy() - self._get_target_xy())

    def obj_target_dist_for_state(self, state):
        obj_pos = state.info
        if self.img:
            state = state[0]
        target_pos = state[4:6]
        return np.linalg.norm(obj_pos - target_pos)

    def get_rgb(self):
        if False:
            return np.zeros((64, 64, 3), dtype = np.float32)

        r1 = self.renderer.render_rgb()
        r2 = self.renderer.render_rgb()

        if False:
            if not np.allclose(r1, r2):
                print('not allclose')

        return r2 / 255.0
