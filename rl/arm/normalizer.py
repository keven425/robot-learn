import numpy as np


class Normalizer():

  def __init__(self, ranges):
    self.ranges = np.array(ranges).astype(np.float32)

  def normalize(self, _in):
    _in = np.array(_in).astype(np.float32)
    low = self.ranges[:, 0]
    high = self.ranges[:, 1]
    out = (_in - low) / (high - low) * 2. - 1.
    return out


if __name__ == '__main__':
  ranges = [
    [0, 180],
    [0, 180],
    [45, 135]
  ]
  normalizer = Normalizer(ranges)
  normalized = normalizer.normalize([
    0, 90, 135
  ])
  print(str(normalized))