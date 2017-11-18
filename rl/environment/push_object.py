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


class PushObjectEnv(utils.EzPickle):

    def __init__(self, frame_skip, max_timestep=3000, log_dir='', seed=None):
        self.frame_skip = frame_skip

        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'push_object.xml')
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=frame_skip)
        self.data = self.sim.data
        self.viewer = mujoco_py.MjViewer(self.sim)
        self.joint_names = list(self.sim.model.joint_names)
        self.joint_addrs = [self.sim.model.get_joint_qpos_addr(name) for name in self.joint_names]
        self.obj_name = 'cube'
        self.endeff_name = 'endeffector'
        self.goal_pos = np.array([0., 0.])
        self.rew_scale = 1.
        self.dist_thresh = 0.01
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.t = 0
        self.max_timestep = max_timestep

        pos_actuators = [actuator for actuator in self.model.actuator_names if 'position' in actuator]
        vel_actuators = [actuator for actuator in self.model.actuator_names if 'velocity' in actuator]
        assert(len(pos_actuators) + len(vel_actuators) == len(self.model.actuator_names))
        self.pos_actuator_ids = [self.model.actuator_name2id(actuator) for actuator in pos_actuators]
        self.vel_actuator_ids = [self.model.actuator_name2id(actuator) for actuator in vel_actuators]
        self.actuator_ids = self.pos_actuator_ids
        self.act_dim = len(self.actuator_ids)

        # compute array: position actuator's joint ranges, in order of self.pos_actuator_ids
        pos_actuators_joints = self.model.actuator_trnid[self.pos_actuator_ids][:, 0]
        self.joint_ranges = [self.model.jnt_range[joint] for joint in pos_actuators_joints]
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
        obj_pos = self.get_body_com(self.obj_name)
        obj_pos_xy = obj_pos[:2]

        # distance between object and goal
        dist_sq = np.sum(np.square(obj_pos_xy - self.goal_pos))
        rew_obj_goal = 0.1 * np.exp(-100. * self.rew_scale * dist_sq)

        # distance between object and robot end-effector
        endeff_pos = self.get_body_com(self.endeff_name)
        dist_sq = np.sum(np.square(endeff_pos - obj_pos))
        rew_endeff_obj = 0.02 * np.exp(-100. * dist_sq)
        reward = rew_obj_goal + rew_endeff_obj

        # reward_ctrl = -np.square(action).mean()
        # reward = rew_obj_goal + reward_ctrl
        done = False
        if self.t > self.max_timestep:
            done = True
        self.t += 1
        return ob, reward, done, dict()


    def get_hidden_ob(self):
        cube_com = self.get_body_com(self.obj_name)[:2]
        # endeff_com = self.get_body_com(self.endeff_name)[:2]
        return np.concatenate([
            cube_com
        ])


    def get_dsq_obj_goal(self):
        obj_pos = self.get_body_com(self.obj_name)
        obj_pos_xy = obj_pos[:2]
        # distance between object and goal
        dsq_obj_goal = np.sum(np.square(obj_pos_xy - self.goal_pos))
        return dsq_obj_goal


    def get_dsq_endeff_obj(self):
        obj_pos = self.get_body_com(self.obj_name)
        endeff_pos = self.get_body_com(self.endeff_name)
        dsq_endeff_obj = np.sum(np.square(endeff_pos - obj_pos))
        return dsq_endeff_obj


    def reset(self, rand_obj_pos, rand_arm_pos):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        self.t = 0
        self.sim.reset()
        ob = self.reset_model(rand_obj_pos, rand_arm_pos)
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


    def start_record_video(self, path=None):
        if self.recording:
            print('record video in progress. calling stop before start.')
            self.stop_record_video()
        self.recording = True
        self.viewer._record_video = True
        self.viewer._hide_overlay = True
        fps = (1 / self.viewer._time_per_render)
        path = path or (self.video_path % self.video_idx)
        self.video_process = Process(target=save_video,
                                     args=(self.viewer._video_queue, path, fps))
        self.video_process.start()


    def stop_record_video(self):
        self.viewer._video_queue.put(None)
        self.video_process.join()
        self.video_idx += 1
        self.recording = False
        print('finished recording video %d' % self.video_idx)

    # ----------------------------

    @property
    def spec(self):
        return None

    def reset_model(self, rand_obj_pos, rand_arm_pos):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        """
        init_qpos = self.init_qpos
        if rand_obj_pos:
            # center around zero, with radius 0.03
            # obj_pos = np.random.uniform(size=[2,]) * 0.3 - 0.15
            radius = 0.075
            angle = np.random.uniform(-math.pi, math.pi)
            x = np.cos(angle) * radius
            y = np.sin(angle) * radius
            obj_pos = np.array([x, y])
        else:
            obj_pos = [0., 0.]
        init_qpos[:2] = obj_pos
        self.set_state(init_qpos, self.init_qvel)

        if rand_arm_pos:
            valid = False
            while not valid:
                bigarm_pos = [np.random.uniform(-.3, .3)]
                arm_pos = np.random.uniform(-1., 1., size=[5,])
                arm_pos = np.concatenate([bigarm_pos, arm_pos])
                # arm_pos = np.array([0., 0.5, 0., 0., 1., 0.])
                arm_pos = self.denormalize_pos(arm_pos)
                init_qpos[-6:] = arm_pos
                self.set_state(init_qpos, self.init_qvel)
                contacting = self.data.ncon > 4
                endeff_pos = self.get_body_com(self.endeff_name)
                endeff_above_plane = (endeff_pos[2] > 0)
                valid = not contacting and endeff_above_plane  # default num contact of cube & plane, in the beginning

            # arm_pos = np.array([0.3, 0.5, 0., 0., 1., 0.])
            # arm_pos = self.denormalize_pos(arm_pos)
            # init_qpos[-6:] = arm_pos
            # self.set_state(init_qpos, self.init_qvel)

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
        n_vel_actuators = len(self.vel_actuator_ids)
        vel_ctrl = np.zeros(shape=[n_vel_actuators])
        # clip by -1, 1
        ctrl = np.clip(ctrl, -1., 1.)
        # scale position control up to joint range
        pos_ctrl = self.denormalize_pos(ctrl)
        self.sim.data.ctrl[self.actuator_ids] = pos_ctrl
        self.sim.data.ctrl[self.vel_actuator_ids] = vel_ctrl  # set velocity to zero for damping
        self.sim.step()
        self.sim.forward()


    def denormalize_pos(self, pos_ctrl):
        low = self.joint_ranges[:, 0]
        high = self.joint_ranges[:, 1]
        pos_ctrl = low + (pos_ctrl + 1.) / 2. * (high - low)
        return pos_ctrl


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
        actuator_pos = self.data.actuator_length[self.pos_actuator_ids]
        actuator_vel = self.data.actuator_velocity[self.vel_actuator_ids]
        # actuator velocity can be out of [-1, 1] range, clip
        # actuator_vel = actuator_vel.clip(-1., 1.)
        # normalize pos
        actuator_pos = self.normalize_pos(actuator_pos)
        cube_com = self.get_body_com("cube")
        return np.concatenate([
            cube_com,
            np.cos(actuator_pos),
            np.sin(actuator_pos),
            actuator_pos
        ])


# Separate Process to save video. This way visualization is
# less slowed down.
def save_video(queue, filename, fps):
    writer = imageio.get_writer(filename, fps=fps)
    while True:
        frame = queue.get()
        if frame is None:
            break
        writer.append_data(frame)
    writer.close()



if __name__ == '__main__':
    env = PushObjectEnv(frame_skip=1)
    for i in range(9999999):
        env.reset(rand_obj_pos=True, rand_arm_pos=True)
        env.render()

    # zeros = np.zeros(shape=[6])
    # ones = np.ones(shape=[6])
    for j in range(3):
        # env.start_record_video()
        # for i in range(3000):
        #     acts = np.random.normal(zeros, ones)
        #     _, _, done, _ = env.step(acts)
        #     env.render()
        #     if done:
        #         env.reset()
        # env.stop_record_video()
        for i in range(1500):
            # env.step([0., 0., 0., 0., 0., 0.])
            env.render()
        for i in range(1500):
            # env.step([1., 1., 1., 1., 1., 1.])
            # env.step([0., 0., 1., 0., 0., 0.])
            env.render()
        # for i in range(1500):
        #     env.step([-1., -1., -1., -1., -1., -1.])
        #     # env.step([0., -1., -1., 0., 0., 0.])
        #     env.render()
