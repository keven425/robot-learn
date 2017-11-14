import numpy as np


class Denormalizer():

  def __init__(self, ranges):
    self.ranges = np.array(ranges).astype(np.float32)

  def denormalize(self, _in):
    _in = np.array(_in).astype(np.float32)
    low = self.ranges[:, 0]
    high = self.ranges[:, 1]
    out = low + (_in + 1.) / 2. * (high - low)
    return out


if __name__ == '__main__':
  ranges = [
    [0, 180],
    [0, 180],
    [45, 135]
  ]
  denormalizer = Denormalizer(ranges)
  denormalized = denormalizer.denormalize([
    0., -.5, .5
  ])
  print(str(denormalized))