import numpy as np
from meta_rand_envs.base import RandomEnv
from gym import utils


class HalfCheetahMixtureEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=0, hfield_mode="gentle", change_mode="location", change_prob=0.005, change_steps=100, termination_possible=False, positive_change_point_basis=20, negative_change_point_basis=-20, change_point_interval=10, state_reconstruction_clip=1000):
        self.change_mode = change_mode
        self.change_prob = change_prob
        self.change_steps = change_steps
        self.termination_possible = termination_possible
        self.changed = False
        self.steps = 0
        self._task = 0
        self.goal_velocity = 0
        self.goal_direction_start = 1
        self.goal_direction = self.goal_direction_start
        self.positive_change_point_basis = positive_change_point_basis
        self.negative_change_point_basis = negative_change_point_basis
        self.change_point_interval = change_point_interval
        self.positive_change_point = np.random.randint(self.positive_change_point_basis,
                                                       self.positive_change_point_basis + self.change_point_interval)
        self.negative_change_point = np.random.randint(self.negative_change_point_basis - self.change_point_interval,
                                                       self.negative_change_point_basis)
        RandomEnv.__init__(self, log_scale_limit, 'half_cheetah.xml', 5, hfield_mode=hfield_mode, rand_params=[])
        utils.EzPickle.__init__(self)

        self._init_geom_rgba = self.model.geom_rgba.copy()

    def _step(self, action):
        # with some probability change goal direction
        if self.change_mode == "time":
            prob = np.random.uniform(0, 1)
            if prob < self.change_prob and self.steps > self.change_steps and not self.initialize and not self.changed:
                self.reset_task(self._task + 1 % (len(self.tasks)))
            if prob < self.change_prob and self.steps > self.change_steps and not self.initialize and self.changed:
                self.reset_task(self._task + 1 % (len(self.tasks)))

        # change direction at some position in the world
        if self.change_mode == "location":
            if self.get_body_com("torso")[0] > self.positive_change_point and not self.initialize and self.goal_direction == 1:
                self.reset_task(self._task + 1 % (len(self.tasks)))
            if self.get_body_com("torso")[0] < self.negative_change_point and not self.initialize and self.goal_direction == -1:
                self.reset_task(self._task + 1 % (len(self.tasks)))
            
        reward, reward_run, reward_ctrl = self.compute_reward(action)

        ob = self._get_obs()
        # compared to gym original, we have the possibility to terminate, if the cheetah lies on the back
        if self.termination_possible:
            state = self.state_vector()
            notdone = np.isfinite(state).all() and state[2] >= -2.5 and state[2] <= 2.5
            done = not notdone
        else:
            done = False
        self.steps += 1
        #return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl, true_task=self.goal_velocity if self._task==1 else -1)
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl, true_task=self._task)


    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ]).astype(np.float32).flatten() #.astype(np.float32).flatten() added to make compatible


    # from pearl rlkit
    '''
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()
    '''

    def reset_model(self):
        # change related
        self.recolor()
        if self._task == 1:
            self.goal_velocity = np.random.uniform(0.0, 3.0)

        # standard
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.type = 1
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -20
        
    def change_goal_direction(self):
        self.goal_direction = -1 * self.goal_direction

        self.recolor()
        self.changed = True
        self.steps = 0

    def change_goal_direction_reset(self):
        # reset changes
        self.goal_direction = self.goal_direction_start

        self.recolor()
        self.changed = False
        self.steps = 0

    def recolor(self):
        if self._task == 0:
            self.model.geom_rgba[:] = self._init_geom_rgba.copy()
        elif self._task == 1:
            geom_rgba = self._init_geom_rgba.copy()
            geom_rgba[1:, :3] = np.array([1, 0, 0])
            self.model.geom_rgba[:] = geom_rgba
        elif self._task == 2:
            geom_rgba = self._init_geom_rgba.copy()
            geom_rgba[1:, :3] = np.array([0, 1, 0])
            self.model.geom_rgba[:] = geom_rgba
        elif self._task == 3:
            geom_rgba = self._init_geom_rgba.copy()
            geom_rgba[1:, :3] = np.array([0, 0, 1])
            self.model.geom_rgba[:] = geom_rgba
        else:
            raise NotImplementedError
        
    def sample_tasks(self, num_tasks):
        tasks = list(range(num_tasks))
        tasks = [0,2]
        return tasks

    def compute_reward(self, action):
        # forward running
        if self._task == 0:
            xposbefore = self.sim.data.qpos[0]
            self.do_simulation(action, self.frame_skip)
            xposafter = self.sim.data.qpos[0]
            reward_run = (xposafter - xposbefore) / self.dt
            reward_ctrl = - 0.1 * np.square(action).sum()
            reward = reward_ctrl * 0.0 + reward_run
            return reward, reward_run, reward_ctrl
        # specific velocity
        if self._task == 1:
            xposbefore = self.sim.data.qpos[0]
            self.do_simulation(action, self.frame_skip)
            xposafter = self.sim.data.qpos[0]
            forward_vel = (xposafter - xposbefore) / self.dt
            reward_run = -1.0 * abs(forward_vel - self.goal_velocity)
            reward_ctrl = 0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 0.0 + reward_run
            return reward, reward_run, reward_ctrl
        # flipping
        if self._task == 2:
            xposbefore = self.sim.data.qpos[2]
            self.do_simulation(action, self.frame_skip)
            xposafter = self.sim.data.qpos[2]
            reward_run = (xposafter - xposbefore) / self.dt
            reward_ctrl = - 0.1 * np.square(action).sum()
            reward = reward_ctrl * 0.0 + reward_run
            return reward, reward_run, reward_ctrl
        # jumping
        if self._task == 3:
            self.do_simulation(action, self.frame_skip)
            reward_run = self.get_body_com("torso").flat[2] - 0.5 * self.sim.data.qpos[2]
            reward_ctrl = 0
            reward = reward_run
            return reward, reward_run, reward_ctrl


if __name__ == "__main__":

    env = HalfCheetahMixtureEnv()
    tasks = env.sample_tasks(40)
    while True:
        env.reset()
        env.set_task(np.random.choice(tasks))
        print(env.model.body_mass)
        for _ in range(2000):
            env.render()
            env.step(env.action_space.sample())  # take a random action
