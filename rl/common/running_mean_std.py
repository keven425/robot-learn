import numpy as np


class RunningMeanStd(object):
  # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
  def __init__(self, shape=(), epsilon=1e-2):
    self._sum = np.zeros(shape)
    self._sumsq = np.ones(shape) * epsilon
    self._count = epsilon


  def update(self, x):
    x = x.astype('float64')
    self._sum += x.sum(axis=0)
    self._sumsq += np.square(x).sum(axis=0)
    self._count += x.shape[0]


  def get_mean_std(self):
    mean = self._sum / self._count
    std = self._sumsq / self._count - np.square(mean)
    std = np.sqrt(np.maximum(std, 1e-2))
    return mean, std


if __name__ == '__main__':
    running_mean_std = RunningMeanStd(shape=[1, 3])
    running_mean_std.update(np.array([
      [1., 2., 3.],
      [2., 3., 4.],
      [3., 4., 5.]
    ]))
    running_mean_std.update(np.array([
      [1., 2., 3.],
      [2., 3., 4.],
      [3., 4., 5.]
    ]))
    mean, std = running_mean_std.get_mean_std()
    print('mean: ' + str(mean))
    print('std: ' + str(std))