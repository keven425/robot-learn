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
from ppo.test import test


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--gpu', action='store_true', help='enable GPU mode', default=False)
    parser.add_argument('--log', help='log directory', type=str, default='')
    parser.add_argument('--load', help='load path of model', type=str, default='')
    parser.add_argument('--test', action='store_true', help='test mode', default=False)
    parser.add_argument('--n_step', help='num rollouts', type=int, default=300)
    parser.add_argument('--n_roll', help='num rollouts', type=int, default=10)
    args = parser.parse_args()
    pp = pprint.PrettyPrinter(indent=1)
    print(pp.pformat(args))
    logger.configure(args.log)
    config = Config()
    env = config.env(frame_skip=config.frame_skip,
                     max_timestep=config.timestep_per_episode,
                     log_dir=args.log,
                     seed=args.seed)
    if args.test:
      test(env, args.gpu, policy=config.policy, load_path=args.load, num_hid_layers=config.num_hid_layers, hid_size=config.hid_size, n_steps=args.n_step, n=args.n_roll)
    else:
      train(env, args.gpu, num_timesteps=config.num_timesteps, seed=args.seed, config=config, log_dir=args.log, load_path=args.load)


def train(env, gpu, num_timesteps, seed, config, log_dir, load_path):
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
                 num_hid_layers=config.num_hid_layers,
                 hid_size=config.hid_size,
                 timesteps_per_batch=config.timesteps_per_batch,
                 clip_param=config.clip_param,
                 beta=config.beta,
                 entcoeff=config.entcoeff,
                 optim_epochs=config.optim_epochs,
                 optim_stepsize=config.optim_stepsize,
                 optim_batchsize=config.optim_batchsize,
                 gamma=config.gamma,
                 lam=config.lam,
                 max_timesteps=num_timesteps,
                 schedule=config.schedule,
                 record_video_freq=config.record_video_freq,
                 log_dir=log_dir,
                 load_path=load_path)
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
