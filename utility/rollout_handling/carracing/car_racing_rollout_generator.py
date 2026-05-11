import os
import numpy as np
from gymnasium.envs.box2d.car_dynamics import Car
from utility.rollout_handling.base_rollout_generator import BaseRolloutGenerator


class RolloutGenerator(BaseRolloutGenerator):
    def __init__(self, config, data_output_dir):
        super().__init__(config, data_output_dir)

    def _standard_rollout(self, environment, thread, current_rollout, rollouts):  # noqa: ARG002
        is_sequence_ok = False
        actions_rollout, states_rollout, reward_rollout, is_done_rollout = [], [], [], []

        while not is_sequence_ok:
            action = [0, 0, 0]
            model = self._get_model() if self.config["data_generator"]['car_racing']["is_ha_agent_driver"] else None
            obs, _ = self._reset(environment)

            for _ in range(self.sequence_length + 1):
                obs, reward, done, info, action = self._step(environment, obs, action, model)
                obs = self._compress_frame(obs, is_resize=True)
                actions_rollout.append(action)
                states_rollout.append(obs)
                reward_rollout.append(reward)
                is_done_rollout.append(done)

            environment.close()
            environment.environment = None  # force re-init on next reset

            is_sequence_ok = len(actions_rollout) >= self.sequence_length
            if not is_sequence_ok:
                actions_rollout, states_rollout, reward_rollout, is_done_rollout = [], [], [], []
                print(f'thread: {thread} - Bad rollout with {len(actions_rollout)} actions - retry...')

        return actions_rollout, states_rollout, reward_rollout, is_done_rollout

    def _reset(self, environment):
        environment.reset()
        core_env = environment.environment.unwrapped
        car_position = np.random.randint(len(core_env.track))
        core_env.car = Car(core_env.world, *core_env.track[car_position][1:4])
        obs, _, _, _, _ = environment.environment.step(np.array([0, 0, 0], dtype=np.float32))
        return obs, car_position

    def _step(self, environment, obs, previous_action, model=None):
        if model:
            z, mu, logvar = model.encode_obs(obs)
            action = model.get_action(z)
        else:
            action = self.action_sampler.sample(previous_action)
        obs, reward, done, info = environment.step(action, ignore_is_done=True)
        return obs, reward, done, info, action

    def _get_model(self):
        from utility.rollout_handling.carracing.ha_implementation.model import Model
        model = Model(load_model=True)
        path = f"{os.getcwd()}/utility/rollout_handling/carracing/ha_implementation/log/carracing.cma.16.64.best.json"
        model.load_model(path)
        return model
