import torch
import numpy as np


def softmax_p(x):
  e_x = torch.exp(x - torch.max(x, dim=1)[0])
  return e_x / torch.sum(e_x, dim=1)


def softmax2d_p(x):
  _max = torch.max(x, dim=-1)[0]
  _max = torch.max(_max, dim=-1)[0]
  _max = _max.view([_max.size()[0], _max.size()[1], 1, 1])
  e_x = torch.exp(x - _max)
  _sum = torch.sum(e_x, dim=-1)
  _sum = torch.sum(_sum, dim=-1)
  _sum = _sum.view([_sum.size()[0], _sum.size()[1], 1, 1])
  return e_x / _sum


def log_prob(act_is, logits):
  act_logp = softmax_logp(logits)
  return torch.gather(act_logp, 1, act_is).view(-1)


def softmax_logp(x):
  # log(e^x1 / (e^x1 + e^x2))
  # x1 - log(e^x1 + e^x2)
  # x1 - x1 - log(1 + e^(x2 - x1))
  # -log(1 + e^(x2 - x1))
  x_numerator = x - torch.max(x, dim=1, keepdim=True)[0]
  e_x = torch.exp(x_numerator)
  return x_numerator - torch.log(torch.sum(e_x, dim=1)).view(-1, 1)


def np_softmax_logp(x):
  # log(e^x2 / (e^x1 + e^x2))
  # x2 - log(e^x1 + e^x2)
  # x2 - x1 - log(e^(x1 - x1) + e^(x2 - x1))
  x_numerator = x - np.max(x, axis=1, keepdims=True)
  e_x = np.exp(x_numerator)
  return x_numerator - np.log(np.sum(e_x, axis=1)).reshape([-1, 1])


if __name__ == '__main__':
  x = np.array([[1., 2.], [2., 3.], [3., 4.]])
  _logp = np_softmax_logp(x)
  print(str(_logp))