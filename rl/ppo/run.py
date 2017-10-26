#!/usr/bin/env python

import bench
import random
import torch
import pprint
import numpy as np
import os.path as osp
import gym, logging
from common import logger
from config import Config


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--gpu', action='store_true', help='enable GPU mode', default=False)
    parser.add_argument('--log', help='log directory', type=str, default='')
    args = parser.parse_args()
    pp = pprint.PrettyPrinter(indent=1)
    print(pp.pformat(args))
    logger.configure(args.log)
    config = Config()
    env = config.env(frame_skip=config.frame_skip,
                     max_timestep=config.timestep_per_episode,
                     log_dir=args.log)
    train(env, args.gpu, num_timesteps=config.num_timesteps, seed=args.seed, config=config)


def train(env, gpu, num_timesteps, seed, config):
    from ppo.ppo_rl import PPO
    set_global_seeds(seed, gpu)
    env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    if hasattr(config, 'wrap_env_fn'):
        env = config.wrap_env_fn(env)
        env.seed(seed)
    ppo_rl = PPO(env,
                 gpu=gpu,
                 policy=config.policy,
                 prob_dist=config.prob_dist,
                 timesteps_per_batch=config.timesteps_per_batch,
                 clip_param=config.clip_param,
                 entcoeff=config.entcoeff,
                 optim_epochs=config.optim_epochs,
                 optim_stepsize=config.optim_stepsize,
                 optim_batchsize=config.optim_batchsize,
                 gamma=config.gamma,
                 lam=config.lam,
                 max_timesteps=num_timesteps,
                 schedule=config.schedule,
                 record_video_freq=config.record_video_freq)
    ppo_rl.run()
    env.close()


def set_global_seeds(seed, gpu):
    torch.manual_seed(seed)
    if gpu:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    main()
