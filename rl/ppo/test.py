import torch
from torch.autograd import Variable



def test(env,
         gpu,
         policy,
         load_path,
         num_hid_layers,
         hid_size):
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy("pi", ob_space, ac_space, hid_size=hid_size, num_hid_layers=num_hid_layers, gpu=gpu)
    pi.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage))
    if gpu:
        pi.cuda()
    pi.train()
    for i in range(3):
        rollout(pi, env, gpu)


def rollout(pi, env, gpu):
    ob = env.reset()
    done = False
    env.env.start_record_video()
    while not done:
        _ob = convert_tensor(ob, gpu)
        ac, vpred = pi.act(_ob, stochastic=False)
        ob, _, done, _ = env.step(ac)
        env.render()
    env.env.stop_record_video()
    env.reset()


def convert_tensor(var, gpu):
    var = Variable(torch.from_numpy(var).float(), requires_grad=False)
    if gpu:
        var = var.cuda()
    return var

