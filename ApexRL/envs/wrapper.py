import gymnasium as gym
import numpy as np
from gymnasium.vector import VectorEnv

from .types import Tensor

class ForwardingVectorWrapper(gym.vector.VectorWrapper):
    """Expose custom attributes defined by the wrapped vector environment."""

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        env = self.__dict__.get("env")
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)


class VectorActionRepeat(ForwardingVectorWrapper):
    """Repeat a batched action until any vector slot finishes."""

    def __init__(self, env: VectorEnv, action_repeat: int = 4) -> None:
        super().__init__(env)
        if action_repeat < 1:
            raise ValueError(f"action_repeat must be positive, got {action_repeat}")
        self._action_repeat = action_repeat

    def step(self, action: Tensor):
        total_reward = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        combined_info = {}

        for _ in range(self._action_repeat):
            obs, reward, step_terminated, step_truncated, info = self.env.step(action)
            total_reward += reward
            terminated |= step_terminated
            truncated |= step_truncated
            combined_info.update(info)
            if np.any(terminated | truncated):
                break

        return obs, total_reward, terminated, truncated, combined_info


class ActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, action_repeat: int) -> None:
        super().__init__(env)
        self.action_repeat = action_repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class ActionClip(ForwardingVectorWrapper):
    """Clip batched actions to the vector environment's action bounds."""

    def step(self, action: Tensor):
        clipped_action = action.clip(
            self.single_action_space.low,
            self.single_action_space.high,
        )
        return self.env.step(clipped_action)


def wrap_vector_env(
    env: VectorEnv,
    *,
    action_repeat: int,
    clip_action: bool,
) -> VectorEnv:
    if clip_action:
        env = ActionClip(env)
    if action_repeat > 1:
        env = VectorActionRepeat(env, action_repeat)
    return env
