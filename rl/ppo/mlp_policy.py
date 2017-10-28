import math
import torch
import torch.nn as nn
import numpy as np
from common.distributions import DiagGaussianPd


# TODO: reuse var b/t policy and value network?
# TODO: normalize ob?
# TODO: use tanh instead of relu?
# TODO: try rnn?
# TODO: try velocity control?

class MlpPolicy(nn.Module):
    recurrent = False
    def __init__(self, name, ob_space, ac_space, hid_size, num_hid_layers, gpu=False):
        super(MlpPolicy, self).__init__()
        self.recurrent = False
        self.name = name
        self.gpu = gpu
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.n_act = ac_space.shape[0]

        n_in = ob_space.shape[0]
        self.fc_values = []
        for i in range(num_hid_layers):
            fc = nn.Linear(n_in, hid_size)
            self.fc_values.append(fc)
            n_in = hid_size
        self.fc_values = nn.ModuleList(self.fc_values)
        self.fc_value = nn.Linear(hid_size, 1)

        n_in = ob_space.shape[0]
        self.fc_acts = []
        for i in range(num_hid_layers):
            fc = nn.Linear(n_in, hid_size)
            self.fc_acts.append(fc)
            n_in = hid_size
        self.fc_acts = nn.ModuleList(self.fc_acts)
        self.fc_act = nn.Linear(hid_size, self.n_act)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        var = np.zeros(shape=[self.n_act])
        self.act_log_stds = torch.nn.Parameter(torch.from_numpy(var).float(), requires_grad=True)
        self.register_parameter('act_log_stds', self.act_log_stds)

        # configure weights
        init_weights_fc(self.fc_value, 1.0)
        init_weights_fc(self.fc_act, 0.01)
        for fc in self.fc_values:
            init_weights_fc(fc, 1.0)
        for fc in self.fc_acts:
            init_weights_fc(fc, 1.0)


    def forward(self, x):
        _x = x
        for fc in self.fc_acts:
            _x = self.tanh(fc(_x))
        act_means = self.tanh(self.fc_act(_x))

        _x = x
        for fc in self.fc_values:
            _x = self.tanh(fc(_x))
        value = self.fc_value(_x).view(-1) # flatten

        act_log_stds = act_means * 0. + self.act_log_stds
        return act_means, act_log_stds, value


    def act(self, ob, stochastic=True):
        act_means, act_log_stds, value = self.forward(ob[None])
        acts = act_means
        if stochastic:
            act_stds = torch.exp(act_log_stds)
            acts = DiagGaussianPd.sample(act_means, act_stds, gpu=self.gpu)
        value = float(value.data.cpu().numpy())
        acts = acts.data.cpu().numpy()[0]
        return acts, value


def init_weights_fc(m, std=1.):
    n = m.in_features + m.out_features
    m.weight.data.normal_(0, std * math.sqrt(2. / n))

