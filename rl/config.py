class Config():

  def __init__(self):
    # atari config
    # import gym
    # from ppo.cnn_policy import CnnPolicy
    # num_frames = 40e6
    # self.env = gym.make(env_id)
    # self.num_timesteps = int(num_frames / 4 * 1.1)
    # self.timesteps_per_batch = 256
    # self.policy = CnnPolicy
    # self.clip_param = 0.2
    # self.entcoeff = 0.01
    # self.optim_epochs = 4
    # self.optim_stepsize = 1e-3
    # self.optim_batchsize = 64
    # self.gamma = 0.99
    # self.lam = 0.95
    # self.schedule = 'linear'
    # self.wrap_env_fn = wrap_train

    # mujoco config
    from environment.push_object import PushObjectEnv
    from ppo.mlp_policy import MlpPolicy
    from common.distributions import DiagGaussianPd
    self.frame_skip = 10
    self.frame_per_episode = 3000
    self.timestep_per_episode = int(self.frame_per_episode / self.frame_skip)
    self.env = PushObjectEnv
    self.prob_dist = DiagGaussianPd
    self.num_hid_layers = 4
    self.hid_size = 64
    # self.recur_timesteps = 32
    self.num_timesteps = 1e20
    self.timesteps_per_batch = self.timestep_per_episode
    self.policy = MlpPolicy
    self.clip_param = 0.2
    self.entcoeff = 0.0
    self.optim_epochs = 10
    self.optim_stepsize = 4e-4
    self.optim_batchsize = self.timestep_per_episode
    self.gamma = 0.99
    self.lam = 0.95
    self.schedule = 'linear'
    self.record_video_freq = 100

def wrap_train(env):
  from common.atari_wrappers import (wrap_deepmind, FrameStack)
  env = wrap_deepmind(env, clip_rewards=True)
  env = FrameStack(env, 4)
  return env