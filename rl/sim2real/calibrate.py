import time
from environment.push_object import PushObjectEnv
from arm.control import Arm


def act(env, command):
  # move both environment and real robot
  ob, _, done, _ = env.step(command)
  env.render()
  # arm.set_positions(command)


if __name__ == '__main__':
  env = PushObjectEnv(frame_skip=1)
  env.reset(rand_init_pos=False)
  # arm = Arm()

  for i in range(1500):
    command = [0., 0., 0., 0., 0., 0.]
    act(env, command)
    time.sleep(0.5)




