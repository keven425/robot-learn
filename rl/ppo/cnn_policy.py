import math
import torch.nn as nn
import numpy as np


class CnnPolicy(nn.Module):
    def __init__(self, name, ob_space, ac_space):
        super(CnnPolicy, self).__init__()
        self.recurrent = False
        self.name = name
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.n_act = self.ac_space.n
        n_channel = ob_space.shape[2]

        self.conv1 = nn.Conv2d(n_channel, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        # compute resulting image size
        size = self.get_size_out(ob_space)
        self.fc1 = nn.Linear(size, 512)
        self.fc_act = nn.Linear(512, self.n_act)
        self.fc_val = nn.Linear(512, 1)
        self.relu = nn.ReLU(inplace=True)

        # configure weights
        init_weights_conv(self.conv1)
        init_weights_conv(self.conv2)
        init_weights_conv(self.conv3)
        init_weights_fc(self.fc1, 1.)
        init_weights_fc(self.fc_act, 0.01)
        init_weights_fc(self.fc_val, 1.)


    def forward(self, x):
        x /= 255.0 # scale down
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # flatten
        x = self.relu(self.fc1(x))
        act_logits = self.fc_act(x)
        value = self.fc_val(x)
        return act_logits, value


    def act(self, ob):
        ob = ob[None]  # add dimension: first axis of size 1
        act_logits, value = self.forward(ob)
        act_p = softmax(act_logits.data.numpy())
        act = np.random.choice(self.n_act, p=act_p[0])
        value = float(value.data.numpy())
        return act, value


    def get_size_out(self, ob_space):
        size_h = ob_space.shape[0] / 8 - 7 / 2
        size_w = ob_space.shape[1] / 8 - 7 / 2
        size = size_h * size_w * 64
        assert (size % 1 == 0.0)  # make sure is int
        size = int(size)
        return size


def init_weights_conv(m):
    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    m.weight.data.normal_(0, math.sqrt(2. / n))


def init_weights_fc(m, std=1.):
    n = m.in_features + m.out_features
    m.weight.data.normal_(0, std * math.sqrt(2. / n))


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1))
    return e_x / e_x.sum(axis=1)