import environment.actions.action_sampler_factory as action_sampler


class BaseEnvironment:
    def __init__(self, config):
        self.config = config
        self.environment = None
        self.game_name = config['game']
        self.action_sampler = action_sampler.get_action_sampler(config)
        self._is_done = False

    def step(self, action, ignore_is_done=False):
        if self.environment is None:
            raise Exception('Cannot call step before reset.')
        if self._is_done and not ignore_is_done:
            raise Exception('Cannot step since game is done. Please call reset.')

        obs, reward, terminated, truncated, info = self.environment.step(action)
        self._is_done = terminated or truncated
        return obs, reward, self._is_done, info

    def reset(self, seed=None):
        self._is_done = False

    def render(self):
        self.environment.render()

    def sample(self):
        return self.action_sampler.sample()

    def close(self):
        self.environment.close()

    def get_current_reward(self):
        return self.environment.unwrapped.reward
