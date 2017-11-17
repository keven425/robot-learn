import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from common.pytorch_util import softmax2d_p


class CnnEncoder(nn.Module):
    def __init__(self, name, image_h, image_w, n_channel, n_out, n_layers=6, gpu=False):
        super(CnnEncoder, self).__init__()
        self.recurrent = False
        self.name = name
        self.n_out = n_out
        self.gpu = gpu
        self.final_n_c = 16

        n_c_out = 32
        conv1 = nn.Conv2d(n_channel, n_c_out, 5, 1, 2)
        conv_res1 = nn.Conv2d(n_channel, n_c_out, 1, 1, 0)
        self.convs = [conv1]
        self.conv_reses = [conv_res1]
        for i in range(n_layers - 2):
            conv = nn.Conv2d(n_c_out, n_c_out, 3, 1, 1)
            conv_res = nn.Conv2d(n_c_out, n_c_out, 1, 1, 0)
            self.convs.append(conv)
            self.conv_reses.append(conv_res)
        conv_final = nn.Conv2d(n_c_out, self.final_n_c, 3, 1, 1)
        conv_res_final = nn.Conv2d(n_c_out, self.final_n_c, 1, 1, 0)
        self.convs.append(conv_final)
        self.conv_reses.append(conv_res_final)
        self.convs = nn.ModuleList(self.convs)
        self.conv_reses = nn.ModuleList(self.conv_reses)
        self.pool = nn.MaxPool2d(2)

        # compute resulting image size
        size = self.get_size_out(image_h, image_w, n_layers)
        self.fc = nn.Linear(size, 32)
        self.fc_out = nn.Linear(32, n_out)
        self.relu = nn.ReLU(inplace=True)

        # configure weights
        for conv in self.convs:
            init_weights_conv(conv)
        for conv in self.conv_reses:
            init_weights_conv(conv)
        init_weights_fc(self.fc, 0.01)
        init_weights_fc(self.fc_out, 0.01)


    def forward(self, x):
        x /= 255.0 # scale down
        for conv, conv_res in zip(self.convs, self.conv_reses):
            _conv = self.relu(conv(x))
            _conv_res = self.relu(conv_res(x))
            x = _conv + _conv_res

        # compute output
        x = x.view(x.size(0), -1)  # flatten
        hid = self.fc(x)
        pred = self.fc_out(hid)
        return pred, hid


    def act(self, image):
        image = image[None]  # add dimension: first axis of size 1
        _, logits = self.forward(image)
        return logits[0]


    def get_size_out(self, image_h, image_w, n_layers):
        size_h = image_h
        size_w = image_w
        # for i in range(n_layers):
        #     size_h //= 2
        #     size_w //= 2
        size = size_h * size_w * 16
        assert (size % 1 == 0.0)  # make sure is int
        size = int(size)
        return size


def init_weights_conv(m):
    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    m.weight.data.normal_(0, math.sqrt(2. / n))


def init_weights_fc(m, std=1.):
    n = m.in_features + m.out_features
    m.weight.data.normal_(0, std * math.sqrt(2. / n))
