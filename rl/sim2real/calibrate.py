import time
import numpy as np
from environment.push_object import PushObjectEnv
from arm.control import Arm


def act(env, arm, command):
  # move both environment and real robot
  # ob, _, done, _ = env.step(command)
  # env.render()
  arm.set_positions(command)
  angles = arm.get_positions()
  print(angles)


if __name__ == '__main__':
  env = PushObjectEnv(frame_skip=1)
  env.reset(rand_init_pos=False)
  arm = Arm()

  for i in range(1500):
    command = np.array([0.5, 0.5, 0., 0., 0., 0.])
    act(env, arm, command)
    time.sleep(0.5)




