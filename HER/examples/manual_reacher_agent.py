import HER.envs
import sys
import gym
from time import sleep
import numpy as np 

from ipdb import set_trace
    
if __name__ == "__main__":
    
    print("Loading %s"%sys.argv[1])
    np.set_printoptions(precision=3)
    env = gym.make(sys.argv[1])
    EVAL_EPISODE = 10
    reward_mat = []

    try:

        # ob = env.reset()
        # print(ob)
        # while True:
        #     env.render(mode = 'human')

        # print("Crossed over")
            
        for l in range(EVAL_EPISODE):
            print("Evaluating:%d"%(l+1))
            done = False
            i =0
            random_r = 0
            ob = env.reset()
            print(ob)

            # print("l joint", env.data.get_joint_qpos("l_gripper_l_finger_joint"))
            # print("r joint", env.data.get_joint_qpos("l_gripper_r_finger_joint"))
            # print("obj initial pos", env.data.get_joint_qpos("box"))

            k =0
            while k<50:
                k += 1
                sleep(0.1)
                env.render(mode='human')
            # set_trace()
            while((not done) and (i<1000)):
                

                # print("l joint", env.data.get_joint_qpos("l_gripper_l_finger_joint"))
                # print("r joint", env.data.get_joint_qpos("l_gripper_r_finger_joint"))
            
                ee_x, ee_y, ee_z, t_x, t_y, t_z = ob
                
                action = np.array([(t_x - ee_x)/0.05, (t_y -ee_y)/0.05, (t_z -ee_z)/0.05])
                # action = [0.,0.]
                # action = env.action_space.sample()
                
                # for checking grasping
                # action = [0., 0., 0.1, -0.5]
                
                ob, reward, done, info = env.step(action)
                print(i, action, ob, reward)
                # print(i, ob, reward, info)
                # print( i, done)    
                i+=1
                sleep(0.1)
                env.render(mode='human')
                random_r += reward


                # set_trace()

            print("num steps:%d, total_reward:%.4f"%(i, random_r))
            
            k = 0
            while k<10:
                k += 1
                sleep(0.1)
                env.render(mode='human')
            reward_mat += [random_r]
        print("reward - mean:%f, var:%f"%(np.mean(reward_mat), np.var(reward_mat)))
    except KeyboardInterrupt:
        action = [0.,0.]
        ob, reward, done, info = env.step(action)
        env.render(mode='human')
        sleep(0.5)

        print("Exiting!")