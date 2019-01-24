import os
import time
from collections import deque
import pickle
import math

from HER.ddpg.ddpg import DDPG
from HER.ddpg.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import HER.common.tf_util as U

from HER import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI

import os.path as osp
from time import sleep

from ipdb import set_trace

def test(env, render_eval, reward_scale, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_eval_steps, batch_size, memory,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50, **kwargs):
    
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    
       
    saver = tf.train.Saver()

    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        ## restore
        restore_dir = osp.join(kwargs["restore_dir"], "model")
        if (restore_dir is not None):
            print('Restore path : ',restore_dir)
            checkpoint = tf.train.get_checkpoint_state(restore_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                
                saver.restore(U.get_session(), checkpoint.model_checkpoint_path)
                print( "checkpoint loaded:" , checkpoint.model_checkpoint_path)
                tokens = checkpoint.model_checkpoint_path.split("-")[-1]
                # set global step
                global_t = int(tokens)
                print( ">>> global step set:", global_t)
            else:
                print(">>>no checkpoint file found")
        
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
                
        # Evaluate.
        eval_episode_rewards = []
        eval_episode_rewards_history = []
        eval_episode_success = []
        for i in range(10):
            print("Evaluating:%d"%(i+1))
            eval_episode_reward = 0.
            eval_obs = eval_env.reset()
            eval_done = False
            
            while(not eval_done):
                eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                
                # print(eval_obs, max_action*eval_action, eval_info)
                if render_eval:
                    eval_env.render()
                    sleep(0.001)
                    
                eval_episode_reward += eval_r

            print("ended",eval_info["done"])
                
            print("episode reward::%f"%eval_episode_reward)
            
            eval_episode_rewards.append(eval_episode_reward)
            eval_episode_rewards_history.append(eval_episode_reward)
            eval_episode_success.append(eval_info["done"]=="goal reached")
            eval_episode_reward = 0.
            
        print("episode reward - mean:%.4f, var:%.4f, success:%.4f"%(np.mean(eval_episode_rewards), np.var(eval_episode_rewards), np.mean(eval_episode_success)))

            


