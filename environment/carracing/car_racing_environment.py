import random
import numpy as np
import gymnasium as gym
from gymnasium.envs.box2d.car_dynamics import Car
from environment.base_environment import BaseEnvironment


class CarRacingEnvironment(BaseEnvironment):
    def __init__(self, config):
        super().__init__(config)
        self.seed = None
        self.is_skip_zoom = self.config['real_environment']['car_racing']['skip_zoom']
        self.is_random_initial_car_position = self.config['real_environment']['car_racing']['random_intial_car_pos']
        self.is_standard_reward = self.config['real_environment']['car_racing']['standardize_reward']

    def step(self, action, ignore_is_done=False):
        state, reward, is_done, info = super().step(action, ignore_is_done)
        if self.is_standard_reward:
            reward = self._standardize_reward(reward)
        return state, reward, is_done, info

    def reset(self, seed=None):
        super().reset()
        if self.environment is None:
            self.environment = gym.make(self.game_name, render_mode="rgb_array")

        self.seed = seed if seed else random.randint(0, 2 ** 31 - 1)
        obs, _ = self.environment.reset(seed=self.seed)
        if self.is_random_initial_car_position:
            obs = self._randomize_car_pos()
        if self.is_skip_zoom:
            obs = self._skip_zoom()
        return obs

    def _skip_zoom(self):
        return [self.environment.step(np.array([0, 0, 0], dtype=np.float32))[0] for _ in range(50)][-1]

    def _randomize_car_pos(self):
        core_env = self.environment.unwrapped
        random_car_position = np.random.randint(len(core_env.track))
        core_env.car = Car(core_env.world, *core_env.track[random_car_position][1:4])
        obs, _, _, _, _ = self.environment.step(np.array([0, 0, 0], dtype=np.float32))
        return obs

    def _standardize_reward(self, reward):
        reward = 3.0 if reward > 3 else reward
        return round(reward, 1)
