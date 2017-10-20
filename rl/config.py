class Config():

  def __init__(self):
    self.timesteps_per_batch = 256
    self.clip_param = 0.2
    self.entcoeff = 0.01
    self.optim_epochs = 4
    self.optim_stepsize = 1e-3
    self.optim_batchsize = 64
    self.gamma = 0.99
    self.lam = 0.95
    self.schedule = 'linear'