import numpy as np
from meta_rand_envs.base import RandomEnv
from gym import utils
import colorsys
from random import shuffle

class HalfCheetahMixtureEnv(RandomEnv, utils.EzPickle):
    def __init__(self, *args, **kwargs):
        self.meta_mode = 'train'
        self.change_mode = kwargs.get('change_mode', 'time')
        self.change_prob = kwargs.get('change_prob', 1.0)
        self.change_steps = kwargs.get('change_steps', 100)
        self.termination_possible = kwargs.get('termination_possible', False)
        self.steps = 0
        self.goal_velocity = 1.0
        self.positive_change_point_basis = kwargs.get('positive_change_point_basis', 10)
        self.negative_change_point_basis = kwargs.get('negative_change_point_basis', -10)
        self.change_point_interval = kwargs.get('change_point_interval', 1)
        self.base_task = 1
        self.task_specification = 1.0
        self.task_variants = kwargs.get('task_variants', ['velocity', 'direction', 'goal', 'jumping', 'flipping'])

        self.positive_change_point = self.positive_change_point_basis + np.random.random() * self.change_point_interval
        self.negative_change_point = self.negative_change_point_basis - np.random.random() * self.change_point_interval

        RandomEnv.__init__(self, kwargs.get('log_scale_limit', 0), 'half_cheetah.xml', 5, hfield_mode=kwargs.get('hfield_mode', 'gentle'), rand_params=[])
        utils.EzPickle.__init__(self)

        self._init_geom_rgba = self.model.geom_rgba.copy()

    def _step(self, action):
        # with some probability change goal direction
        if self.change_mode == "time":
            prob = np.random.uniform(0, 1)
            if prob < self.change_prob and self.steps > self.change_steps and not self.initialize:
                self.change_task()

        # change direction at some position in the world
        if self.change_mode == "location":
            if self.get_body_com("torso")[0] > self.positive_change_point and not self.initialize:
                self.change_task()
                self.positive_change_point = self.positive_change_point + self.positive_change_point_basis + np.random.random() * self.change_point_interval

            if self.get_body_com("torso")[0] < self.negative_change_point and not self.initialize:
                self.change_task()
                self.negative_change_point = self.negative_change_point + self.negative_change_point_basis - np.random.random() * self.change_point_interval
            
        xposbefore = np.copy(self.sim.data.qpos)
        try:
            self.do_simulation(action, self.frame_skip)
        except:
            raise RuntimeError("Simulation error, common error is action = nan")
        xposafter = np.copy(self.sim.data.qpos)
        ob = self._get_obs()

        if self.base_task == 1: #'velocity'
            forward_vel = (xposafter[0] - xposbefore[0]) / self.dt
            reward_run = -1.0 * abs(forward_vel - self.task_specification)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run
            reward = reward * 1.0

        elif self.base_task == 2: # 'direction'
            reward_run = (xposafter[0] - xposbefore[0]) / self.dt * self.task_specification
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run

        elif self.base_task == 3: # 'goal'
            reward_run = xposafter[0] - self.task_specification
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run
            reward = reward / 3.0

        elif self.base_task == 4: # 'flipping'
            reward_run = (xposafter[2] - xposbefore[2]) / self.dt * self.task_specification
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run

        elif self.base_task == 5: # 'jumping'
            reward_run = xposafter[1]
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run
        else:
            raise RuntimeError("bask task not recognized")

        # print(str(self.base_task) + ": " + str(reward))
        # compared to gym original, we have the possibility to terminate, if the cheetah lies on the back
        if self.termination_possible:
            state = self.state_vector()
            notdone = np.isfinite(state).all() and state[2] >= -2.5 and state[2] <= 2.5
            done = not notdone
        else:
            done = False
        self.steps += 1
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl, true_task=dict(base_task=self.base_task, specification=self.task_specification))


    # from pearl rlkit
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    def reset_model(self):
        # reset changepoint
        self.positive_change_point = self.positive_change_point_basis + np.random.random() * self.change_point_interval
        self.negative_change_point = self.negative_change_point_basis - np.random.random() * self.change_point_interval

        # reset tasks
        self.base_task = self._task['base_task']
        self.task_specification = self._task['specification']
        self.recolor()

        # standard
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.type = 1
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -20
        
    def change_task(self):
        if self.meta_mode == 'train':
            task = np.random.choice(self.train_tasks)
            self.base_task = task['base_task']
            self.task_specification = task['specification']
            self.color = task['color']
        elif self.meta_mode == 'test':
            task = np.random.choice(self.test_tasks)
            self.base_task = task['base_task']
            self.task_specification = task['specification']
            self.color = task['color']

        self.recolor()
        self.steps = 0

    def recolor(self):
        geom_rgba = self._init_geom_rgba.copy()
        rgb_value = self.color
        geom_rgba[1:, :3] = np.asarray(rgb_value)
        self.model.geom_rgba[:] = geom_rgba
        
    def sample_tasks(self, num_tasks):
        num_base_tasks = len(self.task_variants)
        num_tasks_per_subtask = int(num_tasks / num_base_tasks)
        num_tasks_per_subtask_half = int(num_tasks_per_subtask / 2)
        np.random.seed(1337)

        tasks = []
        # velocity tasks
        if 'velocity' in self.task_variants:
            velocities = np.random.uniform(1.0, 3.0, size=(num_tasks_per_subtask,))
            tasks_velocity = [{'base_task': 1, 'specification': velocity, 'color': np.array([1,0,0])} for velocity in velocities]
            tasks += (tasks_velocity)

        # direction
        if 'direction' in self.task_variants:
            directions = np.concatenate((np.ones(num_tasks_per_subtask_half), (-1) * np.ones(num_tasks_per_subtask_half)))
            tasks_direction = [{'base_task': 2, 'specification': direction, 'color': np.array([0,1,0])} for direction in directions]
            tasks += (tasks_direction)

        # goal
        if 'goal' in self.task_variants:
            goals = np.random.uniform(-5, 5, size=(num_tasks_per_subtask,))
            tasks_goal = [{'base_task': 3, 'specification': goal, 'color': np.array([0,0,1])} for goal in goals]
            tasks += (tasks_goal)

        # flipping
        if 'flipping' in self.task_variants:
            directions = np.concatenate((np.ones(num_tasks_per_subtask_half), (-1) * np.ones(num_tasks_per_subtask_half)))
            tasks_flipping = [{'base_task': 4, 'specification': direction, 'color': np.array([0.5,0.5,0])} for direction in directions]
            tasks += (tasks_flipping)

        # jumping
        if 'jumping' in self.task_variants:
            tasks_jumping = [{'base_task': 5, 'specification': 0, 'color': np.array([0,0.5,0.5])} for _ in range(num_tasks_per_subtask)]
            tasks += (tasks_jumping)

        #shuffle(tasks)
        return tasks

    def set_meta_mode(self, mode):
        self.meta_mode = mode


if __name__ == "__main__":

    env = HalfCheetahChangingVelEnv()
    tasks = env.sample_tasks(40)
    while True:
        env.reset()
        env.set_task(np.random.choice(tasks))
        print(env.model.body_mass)
        for _ in range(2000):
            env.render()
            env.step(env.action_space.sample())  # take a random action
