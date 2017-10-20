class Config():

  def __init__(self):
    # atari config
    from ppo.cnn_policy import CnnPolicy
    num_frames = 40e6
    self.num_timesteps = int(num_frames / 4 * 1.1)
    self.timesteps_per_batch = 256
    self.policy = CnnPolicy
    self.clip_param = 0.2
    self.entcoeff = 0.01
    self.optim_epochs = 4
    self.optim_stepsize = 1e-3
    self.optim_batchsize = 64
    self.gamma = 0.99
    self.lam = 0.95
    self.schedule = 'linear'
    self.wrap_env_fn = wrap_train

    # mujoco config
    # from ppo.ppo_rl import MlpPolicy
    # self.num_timesteps = 1e6
    # self.timesteps_per_batch = 2048
    # self.policy = MlpPolicy
    # self.clip_param = 0.2
    # self.entcoeff = 0.0,
    # self.optim_epochs = 10
    # self.optim_stepsize = 3e-4
    # self.optim_batchsize = 64
    # self.gamma = 0.99
    # self.lam = 0.95
    # self.schedule = 'linear'

def wrap_train(env):
  from common.atari_wrappers import (wrap_deepmind, FrameStack)
  env = wrap_deepmind(env, clip_rewards=True)
  env = FrameStack(env, 4)
  return env