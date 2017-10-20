#!/usr/bin/env python

import bench
import os.path as osp
import gym, logging
from common import logger
from config import Config


def train(env_id, gpu, num_timesteps, config):
    from ppo.ppo_rl import PPO
    env = gym.make(env_id)
    env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), "monitor.json"))
    gym.logger.setLevel(logging.WARN)
    if hasattr(config, 'wrap_env_fn'):
      env = config.wrap_env_fn(env)
    ppo_rl = PPO(env,
                 gpu=gpu,
                 policy=config.policy,
                 timesteps_per_batch=config.timesteps_per_batch,
                 clip_param=config.clip_param,
                 entcoeff=config.entcoeff,
                 optim_epochs=config.optim_epochs,
                 optim_stepsize=config.optim_stepsize,
                 optim_batchsize=config.optim_batchsize,
                 gamma=config.gamma,
                 lam=config.lam,
                 max_timesteps=num_timesteps,
                 schedule=config.schedule)
    ppo_rl.run()
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--gpu', action='store_true', help='enable GPU mode', default=False)
    parser.add_argument('--log', help='log directory', type=str, default='')
    args = parser.parse_args()
    logger.configure(args.log)
    config = Config()
    train(args.env, args.gpu, num_timesteps=config.num_timesteps, config=config)

if __name__ == '__main__':
    main()
