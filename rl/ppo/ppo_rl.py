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



class PPO(nn.Module):
    def __init__(self,
                 env,
                 gpu,
                 policy,
                 prob_dist,
                 num_hid_layers,
                 hid_size,
                 timesteps_per_batch,  # timesteps per actor per update
                 clip_param,  # clipping parameter epsilon, entropy coeff
                 beta,
                 entcoeff,
                 optim_epochs,  # optimization hypers
                 optim_stepsize,
                 optim_batchsize,
                 gamma,  # advantage estimation
                 lam,
                 max_timesteps=0,  # time constraint
                 max_episodes=0,
                 max_iters=0,
                 max_seconds=0,
                 callback=None,  # you can do anything in the callback, since it takes locals(), globals()
                 adam_epsilon=1e-5,
                 schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
                 record_video_freq=100,
                 log_dir=''
                 ):
        super(PPO, self).__init__()
        self.env = env
        self.prob_dist = prob_dist
        self.gpu = gpu
        self.timesteps_per_batch = timesteps_per_batch
        self.clip_param = clip_param
        self.beta = beta
        self.entcoeff = entcoeff
        self.optim_epochs = optim_epochs
        self.optim_stepsize = optim_stepsize
        self.optim_batchsize = optim_batchsize
        self.gamma = gamma
        self.lam = lam
        self.max_timesteps = max_timesteps
        self.max_episodes = max_episodes
        self.max_iters = max_iters
        self.max_seconds = max_seconds
        self.callback = callback
        self.adam_epsilon = adam_epsilon
        self.schedule = schedule
        self.record_video_freq = record_video_freq
        self.log_dir = log_dir
        self.save_model_path = os.path.join(self.log_dir, 'model.pth')

        # Setup losses and stuff
        # ----------------------------------------
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        self.pi = policy("pi", self.ob_space, self.ac_space, hid_size=hid_size, num_hid_layers=num_hid_layers, gpu=gpu)  # Construct network for new policy
        self.oldpi = policy("oldpi", self.ob_space, self.ac_space, hid_size=hid_size, num_hid_layers=num_hid_layers, gpu=gpu)  # Network for old policy
        if self.gpu:
            self.pi.cuda()
            self.oldpi.cuda()
        # only gradient descent on new policy
        self.optimizer = AdamVariableLr(self.pi.parameters(), lr=self.optim_stepsize, eps=self.adam_epsilon)
        self.loss_names = ["pol_surr", "pol_entpen", "vf_loss", "hid_loss", "kl", "ent"]


    '''
    atarg: Target advantage function (if applicable)
    ret: Empirical return
    lrmult: learning rate multiplier, updated with schedule    
    '''
    def forward(self, ob, hid_ob, ac, atarg, _return, lr_mult):
        self.clip_param = self.clip_param * lr_mult  # Annealed cliping parameter epislon
        act_means_old, act_log_stds_old, value_old, dists_old = self.oldpi.forward(ob)
        act_means_new, act_log_stds_new, value_new, dists_new = self.pi.forward(ob)

        kl_old_new = self.prob_dist.kl(act_means_old, act_means_new, act_log_stds_old, act_log_stds_new)
        _entropy = self.prob_dist.entropy(act_log_stds_new)
        mean_kl = torch.mean(kl_old_new)
        kl_loss = mean_kl * self.beta
        mean_entropy = torch.mean(_entropy)
        pol_entpen = -mean_entropy * self.entcoeff

        act_logp_old = self.prob_dist.log_prob(ac, act_means_old, act_log_stds_old)
        act_logp_new = self.prob_dist.log_prob(ac, act_means_new, act_log_stds_new)
        log_ratio = act_logp_new - act_logp_old
        log_ratio = torch.clamp(log_ratio, max=15)
        ratio = torch.exp(log_ratio)  # pnew / pold
        surr1 = ratio * atarg  # surrogate from conservative policy iteration
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * atarg  #
        pol_surr = -torch.mean(torch.min(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)

        assert(value_new.size() == _return.size())
        vf_loss = torch.mean(torch.pow(value_new - _return, 2))
        hid_loss = torch.mean(torch.pow(dists_new - hid_ob, 2))

        total_loss = pol_surr + pol_entpen + vf_loss + hid_loss + kl_loss
        losses = [pol_surr, pol_entpen, vf_loss, hid_loss, mean_kl, mean_entropy]
        return total_loss, losses


    def run(self):
        # switch to train mode
        self.train()

        # Prepare for rollouts
        seg_generator = self.traj_segment_generator(self.pi, self.env, self.timesteps_per_batch)
        episodes_so_far = 0
        timesteps_so_far = 0
        iters_so_far = 0
        tstart = time.time()
        best_rew = 0.
        lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
        rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
        self.check_time_constraints()

        while True:
            if self.callback: self.callback(locals(), globals())
            if self.max_timesteps and timesteps_so_far >= self.max_timesteps:
                break
            elif self.max_episodes and episodes_so_far >= self.max_episodes:
                break
            elif self.max_iters and iters_so_far >= self.max_iters:
                break
            elif self.max_seconds and time.time() - tstart >= self.max_seconds:
                break
            cur_lrmult = self.get_lr_multiplier(timesteps_so_far)

            logger.log("********** Iteration %i ************"%iters_so_far)

            segment = seg_generator.__next__()
            self.add_vtarg_and_adv(segment, self.gamma, self.lam)

            ob, hid_ob, ac, atarg, tdlamret = segment["ob"], segment["hid_ob"], segment["ac"], segment["adv"], segment["tdlamret"]
            vpredbefore = segment["vpred"] # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
            d = Dataset(dict(ob=ob, hid_ob=hid_ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not self.pi.recurrent)
            optim_batchsize = self.optim_batchsize or ob.shape[0]

            # update running mean/std for policy
            # if hasattr(self.pi, "ob_rms"): self.pi.ob_rms.update(ob)

            # set old parameter values to new parameter values
            self.oldpi.load_state_dict(self.pi.state_dict())

            logger.log("Optimizing...")
            logger.log(fmt_row(13, self.loss_names))
            # Here we do a bunch of optimization epochs over the data
            for _ in range(self.optim_epochs):
                losses = [] # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(self.optim_batchsize):
                    self.optimizer.zero_grad()
                    # batch['ob'] = rearrange_batch_image(batch['ob'])
                    batch = self.convert_batch_tensor(batch)
                    total_loss, *newlosses = self.forward(batch["ob"], batch["hid_ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    total_loss.backward()
                    self.optimizer.step(_step_size=self.optim_stepsize * cur_lrmult)
                    losses.append(torch.stack(newlosses[0], dim=0).view(-1))
                mean_losses = torch.mean(torch.stack(losses, dim=0), dim=0).data.cpu().numpy()
                logger.log(fmt_row(13, mean_losses))

            logger.log("Evaluating losses...")
            losses = []
            for batch in d.iterate_once(self.optim_batchsize):
                # batch['ob'] = rearrange_batch_image(batch['ob'])
                batch = self.convert_batch_tensor(batch)
                _, *newlosses = self.forward(batch["ob"], batch["hid_ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                losses.append(torch.stack(newlosses[0], dim=0).view(-1))
            mean_losses = torch.mean(torch.stack(losses, dim=0), dim=0).data.cpu().numpy()
            logger.log(fmt_row(13, mean_losses))
            for (lossval, name) in zipsame(mean_losses, self.loss_names):
                logger.record_tabular("loss_"+name, lossval)
            logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
            lrlocal = (segment["ep_lens"], segment["ep_rets"]) # local values
            lens, rews = map(flatten_lists, zip(*[lrlocal]))
            mean_rew = np.mean(rews)
            if mean_rew > best_rew:
                torch.save(self.pi.state_dict(), self.save_model_path)
                print('saved model to: ' + self.save_model_path)
                best_rew = mean_rew
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("RewThisIter", mean_rew)
            logger.record_tabular("EpThisIter", len(lens))
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)
            logger.dump_tabular()

            if iters_so_far % self.record_video_freq == 0:
                self.record_video(self.pi, self.env)


    def get_lr_multiplier(self, timesteps_so_far):
        if self.schedule == 'constant':
            cur_lrmult = 1.0
        elif self.schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / self.max_timesteps, 0)
        else:
            raise NotImplementedError
        return cur_lrmult


    def check_time_constraints(self):
        assert (sum([self.max_iters > 0,
                     self.max_timesteps > 0,
                     self.max_episodes > 0,
                     self.max_seconds > 0]) == 1,
                "Only one time constraint permitted")


    def traj_segment_generator(self, pi, env, horizon):
        t = 0
        ac = env.action_space.sample()  # not used, just so we have the datatype
        new = True  # marks if we're on first timestep of an episode
        ob, hid_ob = env.reset(rand_init_pos=True)

        cur_ep_ret = 0  # return in current episode
        cur_ep_len = 0  # len of current episode
        ep_rets = []  # returns of completed episodes in this segment
        ep_lens = []  # lengths of ...

        # Initialize history arrays
        obs = np.array([ob for _ in range(horizon)])
        hid_obs = np.array([hid_ob for _ in range(horizon)])
        rews = np.zeros(horizon, 'float32')
        vpreds = np.zeros(horizon, 'float32')
        news = np.zeros(horizon, 'int32')
        acs = np.array([ac for _ in range(horizon)])
        prevacs = acs.copy()

        while True:
            prevac = ac
            # _ob = rearrange_image(ob)
            _ob = self.convert_tensor(ob)
            ac, vpred = pi.act(_ob, stochastic=True) # TODO: stochastic arg required?
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if t > 0 and t % horizon == 0:
                yield {"ob": obs, "hid_ob": hid_obs, "rew": rews, "vpred": vpreds, "new": news,
                       "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                       "ep_rets": ep_rets, "ep_lens": ep_lens}
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_lens = []
            i = t % horizon
            obs[i] = ob
            hid_obs[i] = hid_ob
            vpreds[i] = vpred
            news[i] = new
            acs[i] = ac
            prevacs[i] = prevac

            ob, hid_ob, rew, new, _ = env.step(ac)
            # env.render()
            rews[i] = rew

            cur_ep_ret += rew
            cur_ep_len += 1
            if new:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                ob, hid_ob = env.reset(rand_init_pos=True)
            t += 1


    def record_video(self, pi, env):
        ob, _ = env.reset(rand_init_pos=True)
        done = False
        env.env.start_record_video()
        while not done:
            _ob = self.convert_tensor(ob)
            ac, vpred = pi.act(_ob, stochastic=False)
            ob, _, _, done, _ = env.step(ac)
            env.render()
        env.env.stop_record_video()
        env.reset(rand_init_pos=True)



    def add_vtarg_and_adv(this, seg, gamma, lam):
        """
        Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
        """
        # last element is only used for last vtarg, but we already zeroed it if last new = 1
        new = np.append(seg["new"], 0)
        vpred = np.append(seg["vpred"], seg["nextvpred"])
        T = len(seg["rew"])
        seg["adv"] = gaelam = np.empty(T, 'float32')
        rew = seg["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[t + 1]
            delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        seg["tdlamret"] = seg["adv"] + seg["vpred"]


    def convert_batch_tensor(self, batch):
        for key in batch.keys():
            batch[key] = self.convert_tensor(batch[key])
        return batch


    def convert_tensor(self, var):
        var = Variable(torch.from_numpy(var).float(), requires_grad=False)
        if self.gpu:
            var = var.cuda()
        return var


# def rearrange_image(ob):
#     return np.transpose(ob, [2, 0, 1])
#
#
# def rearrange_batch_image(ob):
#     return np.transpose(ob, [0, 3, 1, 2])


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


# def kl_divergence(logits1, logits2):
#     a0 = logits1 - torch.max(logits1, dim=-1, keepdim=True)[0]
#     a1 = logits2 - torch.max(logits2, dim=-1, keepdim=True)[0]
#     ea0 = torch.exp(a0)
#     ea1 = torch.exp(a1)
#     z0 = torch.sum(ea0, dim=-1, keepdim=True)
#     z1 = torch.sum(ea1, dim=-1, keepdim=True)
#     p0 = ea0 / z0
#     return torch.sum(p0 * (a0 - torch.log(z0) - a1 + torch.log(z1)), dim=-1)
#
#
# def entropy(logits):
#     a0 = logits - torch.max(logits, dim=-1, keepdim=True)[0]
#     ea0 = torch.exp(a0)
#     z0 = torch.sum(ea0, dim=-1, keepdim=True)
#     p0 = ea0 / z0
#     return torch.sum(p0 * (torch.log(z0) - a0), dim=-1)