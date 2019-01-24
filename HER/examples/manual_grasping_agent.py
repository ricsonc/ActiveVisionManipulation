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

        action1 = [0.,0.0,0., 1.]
        action2 = [0.1,0., -0.5, 1.]
        action3 = [0.,0.,0.,-1.]
        action4 = [0.,0.,0.1,-1.]
        

        for l in range(EVAL_EPISODE):
            print("Evaluating:%d"%(l+1))
            done = False
            i =0
            random_r = 0
            ob = env.reset()
            print(ob)

            # set_trace()
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
                

                # print("l joint", env.env.sim.data.get_joint_qpos("robot0:l_gripper_finger_joint"), env.env.sim.data.ctrl[0])
                # print("r joint", env.env.sim.data.get_joint_qpos("robot0:r_gripper_finger_joint"), env.env.sim.data.ctrl[1])
                
                # print("l joint", env.sim.data.get_joint_qpos("l_gripper_l_finger_joint"), env.sim.data.ctrl[0])
                # print("r joint", env.sim.data.get_joint_qpos("l_gripper_r_finger_joint"), env.sim.data.ctrl[1])
                        
                # if(i<3):
                #     action = action1
                # elif(i<10):
                #     action = action2
                # elif(i<15):
                #     action = action3
                # elif(i<25):
                #     action = action4
                # else:
                #     action = [0.,0.,0.,-1.]
                
                # action = env.action_space.sample()
                # action = [0.,0.,0.,((-1)**(i))]
                ee = ob[:3]
                obj = ob[3:6]
                target = ob[-3:]
                tmp = ((target - obj)/0.05).tolist()
                action = tmp + [-1.]
                
                ob, reward, done, info = env.step(action)
                print(i, action, ob, reward)
                # print(i, ob, reward, info)
                # print( i, action)    
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