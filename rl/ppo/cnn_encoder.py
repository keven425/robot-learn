import math
import torch.nn as nn
import numpy as np


class CnnEncoder(nn.Module):
    def __init__(self, name, image_h, image_w, n_channel, n_out, n_layers=6):
        super(CnnEncoder, self).__init__()
        self.recurrent = False
        self.name = name
        self.n_out = n_out

        n_c_out = 32
        self.conv1 = nn.Conv2d(n_channel, n_c_out, 5, 1, 2)
        self.convs = [self.conv1]
        self.conv_reses = [self.conv1]
        for i in range(n_layers - 1):
            conv = nn.Conv2d(n_c_out, 64, 3, 1, 1)
            conv_res = nn.Conv2d(n_c_out, 64, 1, 1, 0)
            self.convs.append(conv)
            self.conv_reses.append(conv_res)
            n_c_out = 64
        self.convs = nn.ModuleList(self.convs)
        self.conv_reses = nn.ModuleList(self.conv_reses)
        self.pool = nn.MaxPool2d(2)

        # compute resulting image size
        size = self.get_size_out(image_h, image_w, n_layers)
        self.fc1 = nn.Linear(size, 512)
        self.fc_act = nn.Linear(512, n_out)
        self.relu = nn.ReLU(inplace=True)

        # configure weights
        for conv in self.convs:
            init_weights_conv(conv)
        init_weights_fc(self.fc1, 1.)
        init_weights_fc(self.fc_act, 0.01)


    def forward(self, x):
        x /= 255.0 # scale down
        for conv, conv_res in zip(self.convs, self.conv_reses):
            _conv = self.relu(conv(x))
            _conv_res = self.relu(conv_res(x))
            x = _conv + _conv_res
            x = self.pool(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.relu(self.fc1(x))
        logits = self.fc_act(x)
        return logits


    def act(self, image):
        image = image[None]  # add dimension: first axis of size 1
        logits = self.forward(image)[0]
        return logits


    def get_size_out(self, image_h, image_w, n_layers):
        size_h = image_h
        size_w = image_w
        for i in range(n_layers):
            size_h //= 2
            size_w //= 2
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
