import os
import re
import numpy as np
import matplotlib
import logging
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 20000
import matplotlib.pyplot as plt

logger = logging.getLogger("224n_project")

# pretty colors
tableau20 = [(31, 119, 180), (174, 199, 232),
             (44, 160, 44), (152, 223, 138),
             (255, 127, 14), (255, 187, 120),
             (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
  r, g, b = tableau20[i]
  tableau20[i] = (r / 255., g / 255., b / 255.)


def plot_line_mv_avg(filepath, batch_x, epoch_x, loss_ys, perf_ys, title, sub_title, legends, x_label, y_perf_label):

  plt.figure(figsize=(12, 9))

  ax = plt.subplot(111)
  _title = plt.suptitle(title, fontsize=24)
  ps = []
  for i, y in enumerate(loss_ys):
    p, = ax.plot(batch_x, loss_ys[i], lw=1.0, color=tableau20[i+4], alpha=0.3)
    ps.append(p)
  window = int(len(loss_ys[0]) / 20)
  p_avgs = []
  for i, y in enumerate(loss_ys):
    mv_avg = moving_average(loss_ys[i], n=window)
    p, = ax.plot(batch_x, mv_avg, lw=1.0, color=tableau20[i+4], alpha=0.6)
    p_avgs.append(p)

  ax2 = ax.twinx()
  for i, y in enumerate(perf_ys):
    p, = ax2.plot(epoch_x, perf_ys[i], linestyle='-', lw=2.5, color=tableau20[i], alpha=1.0)
    ps.append(p)

  ax.spines["top"].set_visible(False)
  ax.spines["bottom"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["left"].set_visible(False)
  ax2.spines["top"].set_visible(False)
  ax2.spines["bottom"].set_visible(False)
  ax2.spines["right"].set_visible(False)
  ax2.spines["left"].set_visible(False)
  ax.set_xlabel(x_label, fontsize=18)
  ax.set_ylabel('', fontsize=18)
  ax2.set_ylabel(y_perf_label, fontsize=18)
  ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=0.3)
  ax2.yaxis.grid(b=True, which='major', color='gray', linestyle='--', alpha=0.3)
  ax.set_title(sub_title, fontsize=18, y=0.5)
  ax2.set_title('', fontsize=18, y=0.5)
  # ttl = ax.title
  # ttl.set_position([.5, 1.05])
  # ttl2 = ax2.title
  # ttl2.set_position([.5, 1.05])

  ax.get_xaxis().tick_bottom()
  ax.get_yaxis().tick_left()
  ax2.get_yaxis().tick_right()
  plt.tick_params(axis="both", which="both", bottom="off", top="off",
                  labelbottom="on", left="off", right="off", labelleft="off")

  plt.legend(ps, legends, fontsize=18, loc='upper left') \
    .get_frame().set_linewidth(0.0)
  plt.tight_layout()
  fig = plt.gcf()
  fig.subplots_adjust(bottom=0.08, top=0.9)

  plt.savefig(filepath)
  plt.close()
  logger.info('saved fig to: \n' + filepath)


def plot(filepath, xs, lines, legends, title, sub_title='', x_label='', y_label='', plot_line=True, plot_mvavg=False):

  plt.figure(figsize=(12, 9))

  ax = plt.subplot(111)
  plt.suptitle(title, fontsize=24)
  ps = []
  mvavg_alpha = 1.
  line_alpha = 1.
  if plot_mvavg and plot_line:
    line_alpha = .3

  if plot_line:
    for i, line in enumerate(lines):
      p, = ax.plot(xs, line, lw=1.0, color=tableau20[i * 2], alpha=line_alpha)
      ps.append(p)

  if plot_mvavg:
    window = xs.shape[0] // 200
    p_avgs = []
    for i, line in enumerate(lines):
      mv_avg = moving_average(line, n=window)
      p, = ax.plot(xs, mv_avg, lw=1.0, color=tableau20[i * 2], alpha=mvavg_alpha)
      p_avgs.append(p)

  if plot_mvavg:
    ps = p_avgs # attach legent to mvavg line

  ax.spines["top"].set_visible(False)
  ax.spines["bottom"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["left"].set_visible(False)
  ax.set_xlabel(x_label, fontsize=18)
  ax.set_ylabel(y_label, fontsize=18)
  ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=0.3)
  ax.set_title(sub_title, fontsize=18, y=0.5)

  ax.get_xaxis().tick_bottom()
  ax.get_yaxis().tick_left()
  plt.tick_params(axis="both", which="both", bottom="off", top="off",
                  labelbottom="on", left="off", right="off", labelleft="on")

  plt.legend(ps, legends, fontsize=18, loc='upper right') \
    .get_frame().set_linewidth(0.0)
  plt.tight_layout()
  fig = plt.gcf()
  fig.subplots_adjust(bottom=0.08, top=0.9)

  plt.savefig(filepath)
  plt.close()
  logger.info('saved fig to: \n' + filepath)


def moving_average(a, n=10):
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  ret /= n
  ret[:n] = [None] * n # pad beginning with nones
  return ret



def mkdir():
  plots_dir = 'plots'
  os.makedirs(plots_dir, exist_ok=True)
  return plots_dir


def parse_log(log_path):
  import json
  ys = []

  with open(log_path, 'r') as fs:
    next(fs)
    for line in fs:
      _line = json.loads(line)
      y = _line['dist_goal']
      ys.append(y)

  return ys


def set_len(a, _len):
  _a = a[:_len]
  if len(_a) < _len:
    _a = [None] * _len
    _a[0:len(a)] = a
  return _a


if __name__ == "__main__":
  # # reward vs decay
  # title = 'Reward Functions vs. Decay (d)'
  # xs = np.arange(0., 0.25, 0.001)
  # decays = [100., 200., 400., 800.]
  # lines = [(np.exp(-decay * xs * xs) - 1.) for decay in decays]
  # legends = ['decay = ' + str(int(decay)) for decay in decays]
  # plot_one_axis('plot.png', xs, lines, legends, title, x_label='x', y_label='r')

  # reward vs scale
  # title = 'Reward Functions vs. Scale (c)'
  # xs = np.arange(0., 0.25, 0.001)
  # scales = [1., .5, .2, .1]
  # lines = [(scale * np.exp(-100 * xs * xs) - 1.) for scale in scales]
  # legends = ['decay = ' + str(scale) for scale in scales]
  # plot_one_axis('plot.png', xs, lines, legends, title, x_label='x', y_label='r')


  ### 1
  # title =   'Ball Near: Joint Control'
  # log1 = 'logs/ppo_ball_pos_ctrl/monitor.json'
  # log2 = 'logs/ppo_ball_vel_force_ctrl/monitor.json'
  # ys1 = parse_log(log1)
  # ys2 = parse_log(log2)
  # _len = min([len(ys1), len(ys2)])
  # ys1 = set_len(ys1, _len)
  # ys2 = set_len(ys2, _len)
  # xs = np.arange(_len)
  # lines = [ys1, ys2]
  # legends = [
  #   'Joint Position Control',
  #   'Joint Velocity Control'
  # ]
  # plot('ball_joint_control.png', xs, lines, legends, title, x_label='episode', y_label='distance to goal (moving avg)', plot_line=False, plot_mvavg=True)


  ### 2
  title = 'Cube Endeffector Control: Features'
  log1 = 'logs/ppo_ik/monitor.json'
  log2 = 'logs/ppo_ik_simple/monitor.json'
  log3 = 'logs/ppo_near_ik/monitor.json'
  log4 = 'logs/ppo_near_ik_simple/monitor.json'
  ys1 = parse_log(log1)
  ys2 = parse_log(log2)
  ys3 = parse_log(log3)
  ys4 = parse_log(log4)
  _len = min([len(ys1), len(ys2), len(ys3), len(ys4)])
  ys1 = set_len(ys1, _len)
  ys2 = set_len(ys2, _len)
  ys3 = set_len(ys3, _len)
  ys4 = set_len(ys4, _len)
  xs = np.arange(_len)
  lines = [ys1, ys2, ys3, ys4]
  legends = [
    'Cube Far: Full Features',
    'Cube Far: Simple Features',
    'Cube Near: Full Features',
    'Cube Near: Simple Features'
  ]
  plot('cube_endeff_control.png', xs, lines, legends, title, x_label='episode', y_label='distance to goal (moving avg)', plot_line=False, plot_mvavg=True)


  ### 3
  # title = 'Cube Far: Controls'
  # plot_file = 'cube_far_control.png'
  # log1 = 'logs/ppo_pos_ctrl/monitor.json'
  # log2 = 'logs/ppo_vel_force_ctrl/monitor.json'
  # log3 = 'logs/ppo_ik_pos_ctrl/monitor.json'
  # log4 = 'logs/ppo_ik/monitor.json'
  # legends = [
  #   'Joint Position Control',
  #   'Joint Velocity Control',
  #   'End-effector Position Control',
  #   'End-effector Velocity Control']

  ### 4
  # title = 'Cube Near: Controls'
  # plot_file = 'cube_near_control.png'
  # log1 = 'logs/ppo_near_pos_ctrl/monitor.json'
  # log2 = 'logs/ppo_near_vel_force_ctrl/monitor.json'
  # log3 = 'logs/ppo_near_ik_pos_ctrl/monitor.json'
  # log4 = 'logs/ppo_near_ik/monitor.json'
  # legends = [
  #   'Joint Position Control',
  #   'Joint Velocity Control',
  #   'End-effector Position Control',
  #   'End-effector Velocity Control']

  # ys1 = parse_log(log1)
  # ys2 = parse_log(log2)
  # ys3 = parse_log(log3)
  # ys4 = parse_log(log4)
  # _len = min([len(ys1), len(ys2)])
  # ys1 = set_len(ys1, _len)
  # ys2 = set_len(ys2, _len)
  # ys3 = set_len(ys3, _len)
  # ys4 = set_len(ys4, _len)
  # xs = np.arange(_len)
  # lines = [ys1, ys2, ys3, ys4]
  # plot(plot_file, xs, lines, legends, title, x_label='episode', y_label='distance to goal (moving avg)', plot_line=False, plot_mvavg=True)
