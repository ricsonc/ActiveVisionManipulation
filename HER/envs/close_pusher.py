from HER.envs import pusher
import numpy as np 

class BaxterEnv(pusher.BaxterEnv):
    
    def __init__(self, max_len=20, **kwargs):
        super(BaxterEnv, self).__init__(max_len = max_len, **kwargs)

        self.target_range_min += 0.05
        self.target_range_max -= 0.05

    def reset_model(self,
        # define one pose in hand 
        gripper_pos = np.array([0.6 , 0.3 , 0.12]),
        add_noise = False):

        # randomize obj loc
        object_qpos = self.sim.data.get_joint_qpos('box')
        assert object_qpos.shape == (7,)
        object_qpos[:self.space_dim] = self.get_rand()
        object_qpos[2] = 0.0 #otherwise it might drift?

        # fill obj to fully on ground
        object_qpos[3:] = np.array([1., 0.,0.,0.]) #quaternion
        self.data.set_joint_qpos('box', object_qpos)

        while 1:
            target_qpos = self.sim.data.get_joint_qpos('target')            
            assert target_qpos.shape == (7,)
            target_qpos[:self.space_dim] = self.get_rand()
            target_qpos[2] = 0.0
            self.data.set_joint_qpos('target', target_qpos)
            
            if 0.05 <= self.obj_target_dist() <= 0.1:
                break
        
        while 1:
            gripper_pos[:self.space_dim] = self.get_rand()
            d = np.linalg.norm(gripper_pos[:self.space_dim] - object_qpos[:self.space_dim])
            if 0.07 < d < 0.2:
                break

        names = ['box', 'target']
        if 'distractor' in self.base_name:
            names.extend(['distractor%d' % i for i in range(1,4)])
        for name in names:
            self.data.set_joint_qvel(name, np.zeros_like(self.data.get_joint_qvel(name)))

        #print("obj2gripper",np.linalg.norm(gripper_pos[:self.space_dim] - object_qpos[:self.space_dim]))

        # print("obj_pos:",object_qpos[:self.space_dim])
        return super(BaxterEnv, self).reset_model(gripper_pos = gripper_pos,
                                            add_noise = add_noise,
                                            randomize_obj = False) #do not mess with the positions
