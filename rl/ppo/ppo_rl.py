import torch
import torch.nn as nn
from torch.autograd import Variable
from ppo.adam_var_lr import AdamVariableLr
import numpy as np
from common import logger, Dataset, explained_variance, fmt_row, zipsame
from common.pytorch_util import softmax_logp
import time
from collections import deque



class PPO(nn.Module):
    def __init__(self,
                 env,
                 policy,
                 timesteps_per_batch,  # timesteps per actor per update
                 clip_param,  # clipping parameter epsilon, entropy coeff
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
                 schedule='constant'  # annealing for stepsize parameters (epsilon and adam)
                 ):
        super(PPO, self).__init__()
        self.env = env
        self.timesteps_per_batch = timesteps_per_batch
        self.clip_param = clip_param
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

        # Setup losses and stuff
        # ----------------------------------------
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        self.pi = policy("pi", self.ob_space, self.ac_space)  # Construct network for new policy
        self.oldpi = policy("oldpi", self.ob_space, self.ac_space)  # Network for old policy
        self.optimizer = AdamVariableLr(self.parameters(), eps=self.adam_epsilon)
        self.loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    '''
    atarg: Target advantage function (if applicable)
    ret: Empirical return
    lrmult: learning rate multiplier, updated with schedule    
    '''
    def forward(self, ob, ac, atarg, _return, lr_mult):
        self.clip_param = self.clip_param * lr_mult  # Annealed cliping parameter epislon
        act_logits_old, value_old = self.oldpi.forward(ob)
        act_logits_new, value_new = self.pi.forward(ob)

        kl_old_new = kl_divergence(act_logits_old, act_logits_new)
        _entropy = entropy(act_logits_new)
        mean_kl = torch.mean(kl_old_new)
        mean_entropy = torch.mean(_entropy)
        pol_entpen = -mean_entropy * self.entcoeff

        ac_is = ac.view(-1, 1).long()
        act_logp_old = softmax_logp(act_logits_old)
        act_logp_new = softmax_logp(act_logits_new)
        act_logp_old = torch.gather(act_logp_old, 1, ac_is)
        act_logp_new = torch.gather(act_logp_new, 1, ac_is)
        ratio = torch.exp(act_logp_new - act_logp_old)  # pnew / pold
        _atarg = atarg.view(-1, 1)
        surr1 = ratio * _atarg  # surrogate from conservative policy iteration
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * _atarg  #
        pol_surr = -torch.mean(torch.min(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
        vf_loss = torch.mean(torch.pow(value_new - _return, 2))
        total_loss = pol_surr + pol_entpen + vf_loss
        losses = [pol_surr, pol_entpen, vf_loss, mean_kl, mean_entropy]
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

            ob, ac, atarg, tdlamret = segment["ob"], segment["ac"], segment["adv"], segment["tdlamret"]
            vpredbefore = segment["vpred"] # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
            d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not self.pi.recurrent)
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
                    batch['ob'] = rearrange_batch_image(batch['ob'])
                    batch = convert_batch_tensor(batch)
                    total_loss, *newlosses = self.forward(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    total_loss.backward()
                    self.optimizer.step(_step_size=self.optim_stepsize * cur_lrmult)
                    losses.append(torch.stack(newlosses[0], dim=0).view(-1))
                mean_losses = torch.mean(torch.stack(losses, dim=0), dim=0).data.numpy()
                logger.log(fmt_row(13, mean_losses))

            logger.log("Evaluating losses...")
            losses = []
            for batch in d.iterate_once(self.optim_batchsize):
                batch['ob'] = rearrange_batch_image(batch['ob'])
                batch = convert_batch_tensor(batch)
                _, *newlosses = self.forward(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                losses.append(torch.stack(newlosses[0], dim=0).view(-1))
            mean_losses = torch.mean(torch.stack(losses, dim=0), dim=0).data.numpy()
            logger.log(fmt_row(13, mean_losses))

            for (lossval, name) in zipsame(mean_losses, self.loss_names):
                logger.record_tabular("loss_"+name, lossval)
            logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
            lrlocal = (segment["ep_lens"], segment["ep_rets"]) # local values
            lens, rews = map(flatten_lists, zip(*[lrlocal]))
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)
            logger.dump_tabular()


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
        ob = env.reset()

        cur_ep_ret = 0  # return in current episode
        cur_ep_len = 0  # len of current episode
        ep_rets = []  # returns of completed episodes in this segment
        ep_lens = []  # lengths of ...

        # Initialize history arrays
        obs = np.array([ob for _ in range(horizon)])
        rews = np.zeros(horizon, 'float32')
        vpreds = np.zeros(horizon, 'float32')
        news = np.zeros(horizon, 'int32')
        acs = np.array([ac for _ in range(horizon)])
        prevacs = acs.copy()

        while True:
            prevac = ac
            _ob = rearrange_image(ob)
            _ob = convert_tensor(_ob)
            ac, vpred = pi.act(_ob) # TODO: stochastic arg required?
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if t > 0 and t % horizon == 0:
                yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                       "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                       "ep_rets": ep_rets, "ep_lens": ep_lens}
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_lens = []
            i = t % horizon
            obs[i] = ob
            vpreds[i] = vpred
            news[i] = new
            acs[i] = ac
            prevacs[i] = prevac

            ob, rew, new, _ = env.step(ac)
            rews[i] = rew

            cur_ep_ret += rew
            cur_ep_len += 1
            if new:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                ob = env.reset()
            t += 1


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


def convert_batch_tensor(batch):
    for key in batch.keys():
        batch[key] = convert_tensor(batch[key])
    return batch


def convert_tensor(var):
    return Variable(torch.from_numpy(var).float(), requires_grad=False)


def rearrange_image(ob):
    return np.transpose(ob, [2, 0, 1])


def rearrange_batch_image(ob):
    return np.transpose(ob, [0, 3, 1, 2])


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


def kl_divergence(logits1, logits2):
    a0 = logits1 - torch.max(logits1, dim=-1, keepdim=True)[0]
    a1 = logits2 - torch.max(logits2, dim=-1, keepdim=True)[0]
    ea0 = torch.exp(a0)
    ea1 = torch.exp(a1)
    z0 = torch.sum(ea0, dim=-1, keepdim=True)
    z1 = torch.sum(ea1, dim=-1, keepdim=True)
    p0 = ea0 / z0
    return torch.sum(p0 * (a0 - torch.log(z0) - a1 + torch.log(z1)), dim=-1)


def entropy(logits):
    a0 = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    ea0 = torch.exp(a0)
    z0 = torch.sum(ea0, dim=-1, keepdim=True)
    p0 = ea0 / z0
    return torch.sum(p0 * (torch.log(z0) - a0), dim=-1)