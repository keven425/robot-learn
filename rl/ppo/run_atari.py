#!/usr/bin/env python

import bench
import os
import os.path as osp
import gym, logging
from common import logger
from config import Config
from ppo.cnn_policy import CnnPolicy

def wrap_train(env):
    from common.atari_wrappers import (wrap_deepmind, FrameStack)
    env = wrap_deepmind(env, clip_rewards=True)
    env = FrameStack(env, 4)
    return env

def train(env_id, gpu, num_frames, config):
    from ppo.ppo_rl import PPO
    env = gym.make(env_id)
    env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), "monitor.json"))
    gym.logger.setLevel(logging.WARN)
    env = wrap_train(env)
    num_timesteps = int(num_frames / 4 * 1.1)
    ppo_rl = PPO(env,
                 gpu=gpu,
                 policy=CnnPolicy,
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
    os.environ['OPENAI_LOGDIR'] = args.log
    config = Config()
    train(args.env, args.gpu, num_frames=40e6, config=config)

if __name__ == '__main__':
    main()
