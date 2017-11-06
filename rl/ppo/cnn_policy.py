import math
import torch
import torch.nn as nn
import numpy as np
from ppo.cnn_encoder import CnnEncoder
from ppo.mlp_policy import MlpPolicy


class CnnPolicy(nn.Module):
  recurrent = False

  def __init__(self, name, image_dim, ob_space, ac_space, hid_size, num_hid_layers, gpu=False):
    super(CnnPolicy, self).__init__()
    self.recurrent = False
    self.name = name
    self.gpu = gpu
    image_h, image_w, n_channel = image_dim
    obj_com_dim = 3
    obj_pose_dim = 4
    endeff_com_dim = 3
    cnn_n_out = obj_com_dim + obj_pose_dim + endeff_com_dim
    self.cnn_encoder = CnnEncoder(name, image_h, image_w, n_channel, n_out=cnn_n_out, gpu=gpu)

    ob_dim = ob_space.shape[0]
    cnn_n_feat = self.cnn_encoder.final_n_c * 2
    mlp_in_dim = ob_dim + cnn_n_feat  # concat cnn_out and observations
    self.mlp_policy = MlpPolicy(name, mlp_in_dim, ac_space, hid_size, num_hid_layers, gpu=gpu)


  def forward(self, ob):
    image, joint = ob
    cnn_pred, cnn_feat = self.cnn_encoder.forward(image)
    mlp_in = torch.cat([cnn_feat, joint], dim=-1)
    mlp_out = self.mlp_policy.forward(mlp_in)
    return cnn_pred, mlp_out


  def act(self, ob, stochastic=True):
    image, joint = ob
    cnn_out = self.cnn_encoder.act(image)
    mlp_in = torch.cat([cnn_out, joint], dim=-1)
    mlp_out = self.mlp_policy.act(mlp_in, stochastic=stochastic)
    return mlp_out