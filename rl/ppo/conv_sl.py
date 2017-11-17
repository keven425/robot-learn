import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from ppo.adam_var_lr import AdamVariableLr
import numpy as np
from common import logger, Dataset, explained_variance, fmt_row, zipsame
from common.pytorch_util import log_prob
import time
from collections import deque



class ConvPretrain(nn.Module):
    def __init__(self,
                 env,
                 gpu,
                 policy,
                 num_hid_layers,
                 hid_size,
                 optim_epochs,
                 optim_stepsize,
                 optim_batchsize,
                 adam_epsilon=1e-5,
                 schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
                 log_dir='',
                 load_path=''
                 ):
        super(ConvPretrain, self).__init__()
        self.env = env
        self.gpu = gpu
        self.optim_epochs = optim_epochs
        self.optim_stepsize = optim_stepsize
        self.optim_batchsize = optim_batchsize
        self.adam_epsilon = adam_epsilon
        self.schedule = schedule
        self.log_dir = log_dir
        self.save_model_path = os.path.join(self.log_dir, 'model.pth')

        # Setup losses and stuff
        # ----------------------------------------
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        self.pi = policy("pi", self.env.env.image_shape, self.ob_space, self.ac_space, hid_size, num_hid_layers, gpu=gpu)
        self.cnn = self.pi.cnn_encoder # pre-train on this
        if self.gpu:
            self.pi.cuda()
        if load_path:
            print('loading weights from: ' + load_path)
            self.pi.load_state_dict(torch.load(load_path))
        # only gradient descent on new policy
        self.optimizer = AdamVariableLr(self.cnn.parameters(), lr=self.optim_stepsize, eps=self.adam_epsilon)
        self.loss_names = ["loss"]


    '''
    atarg: Target advantage function (if applicable)
    ret: Empirical return
    lrmult: learning rate multiplier, updated with schedule    
    '''
    def forward(self, image, label):
        pred, _ = self.cnn.forward(image)
        loss = torch.mean(torch.pow(pred - label, 2))
        return loss


    def run(self):
        # switch to train mode
        self.train()
        best_loss = 999999.

        while True:
            logger.log("Optimizing...")
            logger.log(fmt_row(13, self.loss_names))
            # Here we do a bunch of optimization epochs over the data
            data = self.collect_data(self.env, self.optim_batchsize)
            d = Dataset(data, shuffle=True)
            for _ in range(self.optim_epochs):
                losses = []
                for batch in d.iterate_once(self.optim_batchsize):
                    self.optimizer.zero_grad()
                    batch = self.convert_batch_tensor(batch)
                    loss = self.forward(batch["image"], batch["label"])
                    loss.backward()
                    self.optimizer.step(_step_size=self.optim_stepsize)
                    losses.append(loss)
                mean_loss = torch.mean(torch.stack(losses, dim=0), dim=0).data.cpu().numpy()
                logger.log(fmt_row(13, mean_loss))

            logger.log("Evaluating losses...")
            data = self.collect_data(self.env, self.optim_batchsize)
            d = Dataset(data, shuffle=True)
            losses = []
            for batch in d.iterate_once(self.optim_batchsize):
                # batch['image'] = rearrange_batch_image(batch['image'])
                batch = self.convert_batch_tensor(batch, train=False)
                loss = self.forward(batch["image"], batch["label"])
                losses.append(loss)
            mean_loss = torch.mean(torch.stack(losses, dim=0), dim=0).data.cpu().numpy()
            logger.log(fmt_row(13, mean_loss))
            for (lossval, name) in zipsame(mean_loss, self.loss_names):
                logger.record_tabular("loss_"+name, lossval)
            _mean_loss = mean_loss[0]
            if _mean_loss < best_loss:
                torch.save(self.pi.state_dict(), self.save_model_path)
                print('saved model to: ' + self.save_model_path)
                best_loss = _mean_loss
            logger.record_tabular("BatchLossMean", _mean_loss)
            logger.dump_tabular()


    def collect_data(self, env, batch_size):
        (image, joint), hid_ob = env.reset(rand_init_pos=True)
        _image = rearrange_image(image)
        images = np.array([_image for _ in range(batch_size)])
        labels = np.array([hid_ob for _ in range(batch_size)])

        for i in range(batch_size):
          ob, hid_ob = env.reset(rand_init_pos=True)
          image, _ = ob
          image = rearrange_image(image)
          label = hid_ob
          images[i] = image
          labels[i] = label

        return dict(image=images, label=labels)


    def convert_batch_tensor(self, batch, train=True):
        for key in batch.keys():
            batch[key] = self.convert_tensor(batch[key], train=train)
        return batch


    def convert_tensor(self, var, train=True):
        var = Variable(torch.from_numpy(var).float(), requires_grad=False, volatile=not train)
        if self.gpu:
            var = var.cuda()
        return var


def rearrange_image(ob):
    return np.transpose(ob, [2, 0, 1])


def rearrange_batch_image(ob):
    return np.transpose(ob, [0, 3, 1, 2])


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
