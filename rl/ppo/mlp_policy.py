import math
import torch
import torch.nn as nn
from torch.autograd import Variable
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
        self.recurrent = True
        self.name = name
        self.gpu = gpu
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.n_act = ac_space.shape[0]
        self.hid_size = hid_size
        self.lstm_value = nn.LSTMCell(hid_size, hid_size)
        self.lstm_act = nn.LSTMCell(hid_size, hid_size)

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
        act_means, act_log_stds, value, self.hidden_act, self.hidden_value = self.forward_step(x, self.hidden_act, self.hidden_value)
        return act_means, act_log_stds, value


    def forward_step(self, x, hidden_act, hidden_value):
        _x = x
        for fc in self.fc_acts:
            _x = self.tanh(fc(_x))
        h_act, c_act = self.lstm_value(_x, hidden_act)
        act_means = self.tanh(self.fc_act(h_act))

        _x = x
        for fc in self.fc_values:
            _x = self.tanh(fc(_x))
        h_value, c_value = self.lstm_value(_x, hidden_value)
        value = self.fc_value(h_value).view(-1) # flatten

        act_log_stds = act_means * 0. + self.act_log_stds
        return act_means, act_log_stds, value, (h_act, c_act), (h_value, c_value)


    # def forward(self, x):
    #     n_timestep = x.size(0)
    #     batch_size = x.size(1)
    #     hidden_value = Variable(torch.zeros(batch_size, self.hid_size), requires_grad=False)
    #     hidden_act = Variable(torch.zeros(batch_size, self.hid_size), requires_grad=False)
    #     for i in range(n_timestep):
    #         act_means, act_log_stds, value, hidden_act, hidden_value = self.forward(x, hidden_value, hidden_act)
    #     return act_means, act_log_stds, value


    # reset hidden state for new roll out
    def reset(self, batch_size=1):
        self.hidden_act = (self.init_hidden(batch_size),
                           self.init_hidden(batch_size))
        self.hidden_value = (self.init_hidden(batch_size),
                             self.init_hidden(batch_size))

    def init_hidden(self, batch_size):
        var = Variable(torch.zeros(batch_size, self.hid_size), requires_grad=False)
        if self.gpu:
            var = var.cuda()
        return var

    def act(self, ob, stochastic=True):
        act_means, act_log_stds, value, self.hidden_act, self.hidden_value = self.forward_step(ob[None], self.hidden_act, self.hidden_value)
        acts = act_means
        if stochastic:
            acts = DiagGaussianPd.sample(act_means, act_log_stds, gpu=self.gpu)
        acts = torch.clamp(acts, -1., 1.)
        value = float(value.data.cpu().numpy())
        acts = acts.data.cpu().numpy()[0]
        return acts, value


def init_weights_fc(m, std=1.):
    n = m.in_features + m.out_features
    m.weight.data.normal_(0, std * math.sqrt(2. / n))

