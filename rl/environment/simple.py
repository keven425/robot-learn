import os
import imageio
import atexit
import math
from multiprocessing import Process, Queue
from gym.spaces import Box
from gym import utils
from gym.utils import seeding
import numpy as np
import mujoco_py


class SimpleEnv(utils.EzPickle):

    def __init__(self, frame_skip, max_timestep=3000, log_dir='', seed=None):
        self.frame_skip = frame_skip

        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'simple.xml')
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=frame_skip)
        self.data = self.sim.data
        self.viewer = mujoco_py.MjViewer(self.sim)
        self.joint_names = list(self.sim.model.joint_names)
        self.joint_addrs = [self.sim.model.get_joint_qpos_addr(name) for name in self.joint_names]
        self.endeff_name = 'endeffector'
        self.dist_thresh = 0.01
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.t = 0
        self.max_timestep = max_timestep

        force_actuators = [actuator for actuator in self.model.actuator_names if 'force' in actuator]
        vel_actuators = [actuator for actuator in self.model.actuator_names if 'velocity' in actuator]
        assert(len(force_actuators) + len(vel_actuators) == len(self.model.actuator_names))
        self.force_actuator_ids = [self.model.actuator_name2id(actuator) for actuator in force_actuators]
        self.vel_actuator_ids = [self.model.actuator_name2id(actuator) for actuator in vel_actuators]
        self.actuator_ids = self.vel_actuator_ids
        self.act_dim = len(self.actuator_ids)

        # compute array: position actuator's joint ranges, in order of self.pos_actuator_ids
        force_actuators_joints = self.model.actuator_trnid[self.actuator_ids][:, 0]
        self.joint_ranges = [self.model.jnt_range[joint] for joint in force_actuators_joints]
        self.joint_ranges = np.array(self.joint_ranges)

        # initial position/velocity of robot and box
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        _ob, _reward, _done, _info = self.step(np.zeros(self.act_dim))
        assert not _done
        self.obs_dim = _ob.size

        bounds = self.model.actuator_ctrlrange[self.actuator_ids].copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = Box(low, high)
        self.reward_range = (-np.inf, np.inf)
        self.seed(seed)

        # set up videos
        self.video_idx = 0
        self.video_path = os.path.join(log_dir, "video/video_%07d.mp4")
        self.video_dir = os.path.abspath(os.path.join(self.video_path, os.pardir))
        self.recording = False
        os.makedirs(self.video_dir, exist_ok=True)
        print('Saving videos to ' + self.video_dir)

        # close on exit
        atexit.register(self.close)


    def __del__(self):
      self.close()


    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.do_simulation(action)
        ob = self._get_obs()
        reward = 0.
        done = False
        return ob, reward, done, dict()


    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        self.t = 0
        self.sim.reset()
        ob = self.reset_model()
        return ob


    def render(self, mode='human', close=False):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        if not close: # then we have to check rendering mode
            modes = self.metadata.get('render.modes', [])
            if len(modes) == 0:
                raise Exception('{} does not support rendering (requested mode: {})'.format(self, mode))
            elif mode not in modes:
                raise Exception('Unsupported rendering mode: {}. (Supported modes for {}: {})'.format(mode, self, modes))
        if close:
            if self.viewer is not None:
                self.viewer.finish()
                self.viewer = None
            return

        if mode == 'rgb_array':
            self.viewer.render()
            data, width, height = self._get_viewer().get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        elif mode == 'human':
            self.viewer.render()


    def close(self):
          """Override _close in your subclass to perform any necessary cleanup.

          Environments will automatically close() themselves when
          garbage collected or when the program exits.
          """
          # _closed will be missing if this instance is still
          # initializing.
          if not hasattr(self, '_closed') or self._closed:
            return

          self._close()
          self._closed = True


    @property
    def spec(self):
        return None

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        """
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()


    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass


    # -----------------------------


    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        state = self.sim.get_state()
        state.qpos[:] = qpos
        state.qvel[:] = qvel
        self.sim.set_state(state)
        self.sim.step()
        self.sim.forward()
        # self.model._compute_subtree()  # pylint: disable=W0212
        # self.model.forward()


    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip


    def do_simulation(self, ctrl):
        # compute end position given velocity control
        # qpos_ctrl = [self.sim.data.qpos[addr] + qvel * self.dt for  (addr, qvel) in zip(self.joint_addrs, ctrl)]
        # print(str(qpos_ctrl))

        # compute velocity control
        n_vel_actuators = len(self.force_actuator_ids)
        force_ctrl = np.zeros(shape=[n_vel_actuators])
        # clip by -1, 1
        vel_ctrl = np.clip(ctrl, -1., 1.)
        # scale position control up to joint range
        self.sim.data.ctrl[self.actuator_ids] = vel_ctrl
        self.sim.data.ctrl[self.force_actuator_ids] = force_ctrl  # set velocity to zero for damping
        self.sim.step()
        self.sim.forward()


    def normalize_pos(self, pos):
        low = self.joint_ranges[:, 0]
        high = self.joint_ranges[:, 1]
        pos = (pos - low) / (high - low) * 2. - 1.
        return pos


    # def _get_viewer(self):
    #     if self.viewer is None:
    #         self.viewer = mujoco_py.MjViewer()
    #         self.viewer.start()
    #         self.viewer.set_model(self.model)
    #         self.viewer_setup()
    #     return self.viewer


    # com: center of mass?
    def get_body_com(self, body_name):
        idx = self.model.body_name2id(body_name)
        return self.data.body_xpos[idx]


    def get_body_comvel(self, body_name):
        idx = self.model.body_name2id(body_name)
        return self.data.xvelp[idx]


    def get_body_xmat(self, body_name):
        idx = self.model.body_name2id(body_name)
        return self.data.body_xmat[idx]


    # def state_vector(self):
    #     return np.concatenate([
    #         self.model.data.qpos.flat,
    #         self.model.data.qvel.flat
    #     ])


    def _get_obs(self):
        actuator_pos = self.data.actuator_length[self.actuator_ids]
        actuator_vel = self.data.actuator_velocity[self.actuator_ids]
        # actuator velocity can be out of [-1, 1] range, clip
        # actuator_vel = actuator_vel.clip(-1., 1.)
        # normalize pos
        actuator_pos = self.normalize_pos(actuator_pos)
        return np.concatenate([
            np.cos(actuator_pos),
            np.sin(actuator_pos),
            actuator_pos
        ])



def get_joint_angles_ik(d_endeff):
    jac = env.data.get_body_jacp('endeffector')
    jac = jac.reshape((3, 2))
    _lambda_sq = .0001
    j_jt = jac.dot(jac.T)
    inv = np.linalg.inv(j_jt + _lambda_sq * np.eye(3, 3))
    jacq_inv = jac.T.dot(inv)
    # jacq_inv = np.linalg.inv(jac.T.dot(jac)).dot(jac.T)
    # jacq_inv = jac.T
    d_joints = jacq_inv.dot(d_endeff)
    print(d_joints)
    max = np.abs(d_joints).max()
    if max > 1.:
      d_joints = d_joints / max
    else:
      print('d_joints are all zero')
    return d_joints


if __name__ == '__main__':
    env = SimpleEnv(frame_skip=10)
    env.reset()
    env.data.qpos[:] = [-1, 2]
    env.sim.step()
    env.sim.forward()
    for j in range(100):
        for i in range(100):
            # env.sim.forward()
            d_joints = get_joint_angles_ik([0., -1., 0.])
            # qpos = env.data.qpos + d_joints
            # env.data.qpos[:] = qpos
            # # env.data.qvel[:] = d_joints
            # env.sim.step()
            env.step(d_joints)
            env.render()
        for i in range(100):
            d_joints = get_joint_angles_ik([0., 1., 0.])
            # qpos = env.data.qpos + d_joints
            # env.data.qpos[:] = qpos
            # # env.data.qvel[:] = d_joints
            # env.sim.step()
            # env.sim.forward()
            env.step(d_joints)
            env.render()
