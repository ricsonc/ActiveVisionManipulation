import argparse
import time
import os
import logging
from HER import logger
from HER.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import HER.ddpg.training as training
from HER.ddpg.models import *
from HER.ddpg.memory import Memory
from HER.ddpg.noise import *

import gym
import tensorflow as tf
from mpi4py import MPI
## my imports
import HER.envs

def run(env_id, seed, noise_type, layer_norm, evaluation, memory_size, single_train, factor, static, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Create envs.
    env = gym.make(env_id)
    logger.info("Env info")
    logger.info(env.__doc__)
    logger.info("-"*20)
    gym.logger.setLevel(logging.WARN)

    if evaluation and rank==0:
        if kwargs['eval_env_id']: 
            eval_env_id = kwargs['eval_env_id']
        else: 
            eval_env_id = env_id
        eval_env = gym.make(eval_env_id)
        # del eval_env_id from kwargs
        del kwargs['eval_env_id']
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'epsnorm' in current_noise_type:
            _, stddev, epsilon  = current_noise_type.split('_')
            action_noise = EpsilonNormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions), epsilon= float(epsilon))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    ospace = env.observation_space
    has_image = (not hasattr(ospace, 'shape')) or (not ospace.shape)

    if has_image:
        assert isinstance(env.observation_space, gym.spaces.Tuple)
        env.observation_space.shape = [x.shape for x in env.observation_space.spaces]

    if rank == 0 or not single_train:
        memory = Memory(
            limit=memory_size,
            action_shape=env.action_space.shape,
            observation_shape=env.observation_space.shape
        )
    else:
        memory = None

    if has_image:
        ignore = False
        if ignore:
            critic = IgnoreDepthCritic(layer_norm=layer_norm)
            actor = IgnoreDepthActor(nb_actions, layer_norm=layer_norm, static = static)
        else:
            critic = DepthCritic(layer_norm=layer_norm)
            if factor == 1:
                actor = FactoredDepthActor(nb_actions, layer_norm=layer_norm, static = static)
            elif factor == 0:
                actor = DepthActor(nb_actions, layer_norm=layer_norm, static = static)
            elif factor == 2:
                actor = ImgDepthActor(nb_actions, layer_norm=layer_norm, static = static)
            else:
                print('bad value of factor')
                actor = None
    else:
        critic = Critic(layer_norm=layer_norm)
        actor = Actor(nb_actions, layer_norm=layer_norm, static = static)
        

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(4)
        
    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    training.train(
        env=env, eval_env=eval_env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory,
        single_train = single_train, **kwargs
    )
    
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='Baxter-v1')
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'static', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=0.)
    parser.add_argument('--batch-size', type=int, default=128)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=200)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=40)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=800)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='epsnorm_0.01_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--num-timesteps', type=int, default=None)
    parser.add_argument('--single-train', type=int, default=0)
    boolean_flag(parser, 'evaluation', default=True)
    parser.add_argument('--eval-env-id', type=str, default=None)
    
    parser.add_argument('--memory-size', type=int, default=50000)
    parser.add_argument('--factor', type=int, default=0)
    
    ## saving and restoring param parser
    parser.add_argument('--log-dir', type=str, default='/tmp/her')
    parser.add_argument('--save-freq', type=int, default=1)
    parser.add_argument('--restore-dir', type=str, default=None)
    boolean_flag(parser, 'dologging', default=True)
    boolean_flag(parser, 'invert-grad', default=False)
    boolean_flag(parser, 'her', default=True)
    boolean_flag(parser, 'actor-reg', default=True)
    boolean_flag(parser, 'tf-sum-logging', default=False)

    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.ERROR)
    args = parse_args()
   
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure(dir=args["log_dir"])
        
        logger.info(str(args))
        
    # Run actual script.
    try:
        run(**args)
    except KeyboardInterrupt: 
        print("Exiting!")
