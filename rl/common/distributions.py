import torch
import numpy as np
from torch.autograd import Variable


class DiagGaussianPd():
  @staticmethod
  def log_prob(x, mean, log_std):
    std = torch.exp(log_std)
    term1 = 0.5 * torch.sum(torch.pow((x - mean) / std, 2), dim=-1)
    term2 = 0.5 * np.log(2.0 * np.pi) * x.size(-1)
    term3 = torch.sum(log_std, dim=-1)
    return -(term1 + term2 + term3)


  @staticmethod
  def kl(mean1, mean2, log_std1, log_std2):
    std1 = torch.exp(log_std1)
    std2 = torch.exp(log_std2)
    numerator = torch.pow(std1, 2) + torch.pow(mean1 - mean2, 2)
    denom = 2.0 * torch.pow(std2, 2)
    return torch.sum(log_std2 - log_std1 + numerator / denom - 0.5, dim=-1)


  @staticmethod
  def entropy(log_std):
    return torch.sum(log_std + .5 * np.log(2.0 * np.pi * np.e), dim=-1)


  @staticmethod
  def sample(mean, log_std):
    std = torch.exp(log_std)
    zero_mean = torch.from_numpy(np.zeros(shape=mean.size())).float()
    one_std = torch.from_numpy(np.ones(shape=mean.size())).float()
    normal = Variable(torch.normal(zero_mean, one_std), requires_grad=False)
    return mean + torch.mul(std, normal)
