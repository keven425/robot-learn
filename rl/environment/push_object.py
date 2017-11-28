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
from arm.denormalizer import Denormalizer


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
        self.radiuses = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.075]
        self.level = len(self.radiuses)
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
        self.act_dim = 7

        # compute array: position actuator's joint ranges, in order of self.pos_actuator_ids
        force_actuators_joints = self.model.actuator_trnid[self.actuator_ids][:, 0]
        self.joint_ranges = [self.model.jnt_range[joint] for joint in force_actuators_joints]
        self.joint_ranges = np.array(self.joint_ranges)

        # ranges for endeffector control
        endeff_ranges = [
            [-.15, .15],
            [-.15, .15],
            [0., .4],
            # quaternions
            [-1, 1.],
            [-1, 1.],
            [-1, 1.],
            [-1, 1.]
            # euler angles
            # [-math.pi / 2., math.pi / 2.],
            # [0., math.pi / 2],
            # [-math.pi / 2., math.pi / 2.]
        ]
        self.endeff_denorm = Denormalizer(endeff_ranges)

        # initial position/velocity of robot and box
        self.init_qpos = self.data.qpos.ravel().copy()
        # self.init_qpos[-6:] = [0., .4, 1.7, 0., 1., 0.] # initial position w/ endeffector close to cube
        self.init_qvel = self.data.qvel.ravel().copy()
        _ob, _reward, _done, _info = self.step(np.zeros(self.act_dim))
        assert not _done
        self.obs_dim = _ob.size

        # bounds = self.model.actuator_ctrlrange[self.actuator_ids].copy()
        low = np.array([-1.] * self.act_dim)
        high = np.array([1.] * self.act_dim)
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
        # transform from endeff vel to joint vel
        joint_vels = self.action_to_joint_vel(action)
        self.do_simulation(joint_vels)
        ob = self._get_obs()
        endeff_com = self.get_body_com(self.endeff_name)
        obj_pos = self.get_body_com(self.obj_name)
        obj_pos_xy = obj_pos[:2]

        # distance between object and goal
        dist_sq_og = np.sum(np.square(obj_pos_xy - self.goal_pos))
        rew_obj_goal = 0.1 * (np.exp(-800. * dist_sq_og) - 1.)

        # distance between object and robot end-effector
        endeff_pos = self.get_body_com(self.endeff_name)
        dist_sq_eo = np.sum(np.square(endeff_pos - obj_pos))
        rew_endeff_obj = 0.05 * (np.exp(-50. * dist_sq_eo) - 1.)

        # penalty for nearing singularity
        reward_ctrl = 0.02 * (np.exp(-ik_norm) - 1.)

        reward = rew_obj_goal + rew_endeff_obj + reward_ctrl
        done = False
        info = dict()
        if self.t > self.max_timestep:
            done = True
            info['dist_goal'] = np.sqrt(dist_sq_og)
        self.t += 1
        return ob, reward, done, info

    def action_to_joint_vel(self, action):
        # clip actions
        action = np.clip(action, -1., 1.)
        # denormalize
        action = self.endeff_denorm.denormalize(action)
        action = np.array(action)
        endeff_com = self.get_body_com(self.endeff_name)
        # compute d_pos
        d_pos = action[:3] - endeff_com
        d_pos /= self.dt
        # compute d_quat
        quat = self.get_body_quat(self.endeff_name)
        quat_targ = action[3:]
        quat_targ = self.norm_quat(quat_targ)
        d_quat = quat_targ - quat
        d_quat /= self.dt
        # compute Er, Er inverse
        q0, q1, q2, q3 = quat
        H = np.array([[-q1, q0, -q3, q2],
                      [-q2, q3, q0, -q1],
                      [-q3, -q2, q1, q0]])
        Er_inv = H.T * .5
        # Er = H * 2.
        # d_quat = Er.dot(d_quat)
        joint_vels = self.get_joint_vels_ik(d_pos, d_quat, Er_inv)
        return joint_vels

    def norm_quat(self, quat):
        quat = quat.astype(np.float64)
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 0:
            quat /= quat_norm
        return quat

    def get_joint_vels_ik(self, d_pos, d_quat, Er_inv):
        d_endeff = np.concatenate([d_pos, d_quat], axis=-1)

        # compute jacobian w.r.t. position
        jacp = self.data.get_body_jacp(self.endeff_name)
        jacp = jacp.reshape((3, -1))
        jacp = jacp[:, -6:]

        # compute jacobian w.r.t. quaternion
        jacr = self.data.get_body_jacr(self.endeff_name)
        jacr = jacr.reshape((3, -1))
        jacr = jacr[:, -6:]
        jac_quat = Er_inv.dot(jacr)
        jac = np.concatenate([jacp, jac_quat], axis=0)

        # pseudo inverse of jacobian
        _lambda_sq = .0001
        j_jt = jac.dot(jac.T)
        inv = np.linalg.inv(j_jt + _lambda_sq * np.eye(7, 7))
        jacq_inv = jac.T.dot(inv)
        # jacq_inv = np.linalg.inv(jac.T.dot(jac)).dot(jac.T)
        # jacq_inv = jac.T

        # compute joint velocity
        d_joints = jacq_inv.dot(d_endeff)
        l1_norm = np.square(d_joints).mean()
        
        # normalize
        max = np.abs(d_joints).max()
        if max > 1.:
            d_joints = d_joints / max
        return d_joints, l1_norm


    def reset(self, rand_init_pos=False):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        self.t = 0
        self.sim.reset()
        ob = self.reset_model(rand_init_pos)
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

    def reset_model(self, rand_init_pos):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        """
        init_qpos = self.init_qpos
        if rand_init_pos:
            # center around zero, with radius 0.03
            # obj_pos = np.random.uniform(size=[2,]) * 0.3 - 0.15
            max_radius = self.radiuses[self.level - 1]
            radius = np.random.uniform(0., max_radius)
            print('level: %d, max_radius: %f, radius: %f' % (self.level, max_radius, radius))
            angle = np.random.uniform(-math.pi, math.pi)
            x = np.cos(angle) * radius
            y = np.sin(angle) * radius
            obj_pos = np.array([x, y])
        else:
            obj_pos = [0., 0.]
        init_qpos[:2] = obj_pos
        self.set_state(init_qpos, self.init_qvel)
        return self._get_obs()


    def level_up(self):
        self.level += 1
        n_levels = len(self.radiuses)
        self.level = np.minimum(self.level, n_levels)
        print('increasing level to: %d' % self.level)


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


    def get_body_comvelp(self, body_name):
        idx = self.model.body_name2id(body_name)
        return self.data.body_xvelp[idx]

    def get_body_comvelr(self, body_name):
        idx = self.model.body_name2id(body_name)
        return self.data.body_xvelr[idx]


    def get_body_xmat(self, body_name):
        idx = self.model.body_name2id(body_name)
        return self.data.body_xmat[idx].reshape([3, 3])


    def get_body_quat(self, body_name):
        idx = self.model.body_name2id(body_name)
        return self.data.body_xquat[idx]


    # def state_vector(self):
    #     return np.concatenate([
    #         self.model.data.qpos.flat,
    #         self.model.data.qvel.flat
    #     ])


    def _get_obs(self):
        actuator_pos = self.data.actuator_length[self.actuator_ids]
        # normalize pos
        pos_cos = np.cos(actuator_pos)
        pos_sin = np.sin(actuator_pos)
        actuator_pos_normed = self.normalize_pos(actuator_pos)
        actuator_vel = self.data.actuator_velocity[self.actuator_ids]
        cube_com = self.get_body_com(self.obj_name)
        cube_vel = self.get_body_comvel(self.obj_name)
        endeff_com = self.get_body_com(self.endeff_name)
        endeff_vel = self.get_body_comvel(self.endeff_name)

        return np.concatenate([
            pos_cos,
            pos_sin,
            actuator_pos_normed,
            actuator_vel,
            cube_com,
            cube_vel,
            endeff_com,
            endeff_vel
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


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    euler = np.array([x, y, z])
    return euler


def quaternion_to_euler_angle(quat):
    w, x, y, z = quat
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.atan2(t3, t4)

    return [X, Y, Z]


if __name__ == '__main__':
    env = PushObjectEnv(frame_skip=10)
    env.reset()
    for j in range(100):
        for i in range(200):
            # first three elements are position velocities, last three elements are rotation velocities
            actions = [0., 0., -.5, 0., 0., 1., 0.]
            _, rew, _, _ = env.step(actions)
            env.render()
            # print(rew)
        for i in range(200):
            actions = [0., 0., .3, 0., 0., -1., 0.]
            _, rew, _, _ = env.step(actions)
            env.render()
            # print(rew)
