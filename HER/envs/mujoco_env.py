import os
import os.path as osp
import numpy as np

from gym.utils import seeding
import gym
from gym.envs.robotics.utils import reset_mocap2body_xpos, mocap_set_action, reset_mocap_welds
import socket

try:
    import mujoco_py
    from mujoco_py.mjviewer import MjViewerBasic, MjViewer
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, num_substeps=75):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = osp.join(osp.dirname(__file__), "assets", model_path)
        if not osp.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        else:
            print("Loading model %s"%osp.basename(model_path))
        
        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=num_substeps)
        self.data = self.sim.data
        self.viewer = None

        #due to some sort of weird rendering bug
        self.viewer_throwaway = MjViewerBasic(self.sim) if socket.gethostname() == 'ergo' else None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()

        # limits of motion with some safety limits
        self.target_range_min = np.array([0.3, 0.0, 0.08]) + 0.05
        self.target_range_max = np.array([0.8, 0.6, 0.25]) - 0.05
        
        self.initial_qpos = {
            'left_s0': -0.08,
            'left_s1': -1.0,
            'left_e0': -1.19,
            'left_e1': 1.94,
            'left_w0': 0.67,
            'left_w1': 1.03,
            'left_w2':-0.5
        }

    

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def get_mocap_state(self):
        assert self.sim.data.mocap_pos.shape[0] == 1
        return self.sim.data.mocap_pos.copy()[0, :self.space_dim]

    def reset(self):
        for name, value in self.initial_qpos.items():
            self.data.set_joint_qpos(name, value)
        reset_mocap_welds(self.sim)
        self.sim.forward()
        
        # not sure if this is required. therefore not using currently
        # self.sim.reset()

        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    # def set_state(self, qpos, qvel):
    #     assert qpos.shape == (self.model.model.nq,) and qvel.shape == (self.model.model.nv,)
    #     self.model.data.qpos = qpos
    #     self.model.data.qvel = qvel
    #     self.model.forward()

    @property
    def dt(self):
        return self.sim.model.opt.timestep

    def do_simulation(self, ctrl = None, n_frames=1):
        if ctrl is not None:
            self.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()
        
    def set_cam_position(self, cam_pos):
        for i in range(3):
            self.sim.model.stat.center[i] = cam_pos[i]
        self.sim.model.stat.extent = cam_pos[3]

    def render(self, mode='human', close=False, writer = None, **kwargs):
        if writer:
            output = self.sim.render(width=256, height=256, camera_name = 'cam4')
            writer.append_data(output)
            return
        
        if close:
            if self.viewer is not None:
            #     self._get_viewer().finish()
                self.viewer = None
            return

        if mode == 'rgb_array':
            return self.sim.render(**kwargs)
        elif mode == 'human':
            self._get_viewer().render()

    def _get_viewer(self, mode='human'):
        if self.viewer is None:
            if mode == 'human':
                self.viewer_setup()
                self.viewer = MjViewerBasic(self.sim)
                
        return self.viewer

    def _get_distractor_xy(self):
        return self._get_joint('distractor')
    
    def _get_obj_xy(self):
        return self._get_joint('box') #must be joint, not site

    def _get_obj_for_obs(self): #this is used when..?
        return self._get_obj_xy()
    
    def _get_target_xy(self):
        return self._get_joint('target')

    def _get_ee_xy(self):
        return self._get_site('grip')

    def _get_site(self, name):
        return self.sim.data.get_site_xpos(name)[:self.space_dim]

    def _get_joint(self, name, full = False):
        rval = self.data.get_joint_qpos(name)
        if not full:
            rval = rval[:self.space_dim]
        return rval

    def _set_joint(self, name, state):
        return self.data.set_joint_qpos(name, state)
    
    def get_rand(self, size = None):
        if size is None:
            size = self.space_dim
        return self.np_random.uniform(
            self.target_range_min[:size],
            self.target_range_max[:size],
            size=size
        )

    def __render(self, **kwargs):
        return self.sim.render(**kwargs)

    def rand_state(self):
        pass
