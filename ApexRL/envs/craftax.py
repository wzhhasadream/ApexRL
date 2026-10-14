from __future__ import annotations

from functools import partial
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name
from gymnasium.vector.utils import batch_space

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")  # type: ignore
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)  # type: ignore
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)  # type: ignore


CRAFTAX_ENVS = {
    "Craftax-Classic-Symbolic-v1",
    "Craftax-Classic-Pixels-v1",
    "Craftax-Symbolic-v1",
    "Craftax-Pixels-v1"
}



class CraftaxEnv(gym.vector.VectorEnv):
    """Batched Craftax with Gymnasium SameStep autoreset and NumPy outputs.

    Symbolic observations keep their native shape. Pixels use [N, 1, H, W, C]
    uint8 observations for ApexRL's CNNs; the frame axis contains one frame.
    """

    metadata = {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP, "render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, env_name: str, seed: int, num_envs: int, max_episode_steps: int | None = None, render_mode: str | None = None,
                 use_optimistic_resets: bool = False, optimistic_reset_ratio: int = 16, log_achievements: bool = False) -> None:
        if optimistic_reset_ratio < 1:
            raise ValueError("optimistic_reset_ratio must be positive")
        self.num_reset_envs = (num_envs + optimistic_reset_ratio - 1) // optimistic_reset_ratio if use_optimistic_resets else num_envs
        if "Pixels" in env_name or render_mode == "rgb_array":
            cache_path = "/tmp/apexrl-craftax-classic-texture_cache.pbz2" if env_name.startswith("Craftax-Classic-") else "/tmp/apexrl-craftax-texture_cache.pbz2"
            if env_name.startswith("Craftax-Classic-"):
                from craftax.craftax_classic import constants
            else:
                from craftax.craftax import constants
            constants.TEXTURE_CACHE_FILE = cache_path
        self.env = make_craftax_env_from_name(env_name, auto_reset=False)
        self.params = self.env.default_params
        if max_episode_steps is not None:
            self.params = self.params.replace(max_timesteps=max_episode_steps)
        # Craftax combines time-limit and natural endings into one done flag.
        # Increase the limit by one so the wrapper can classify time limits as truncations.
        self.natural_params = self.params.replace(max_timesteps=self.params.max_timesteps + 1)
        self.num_envs = num_envs
        self.render_mode = render_mode
        self.rng = jax.random.PRNGKey(seed)
        self.states = None

        obs_space = self.env.observation_space(self.params)
        self.pixel_obs = len(obs_space.shape) == 3
        if self.pixel_obs:
            self.single_observation_space = gym.spaces.Box(0, 255, (1, *obs_space.shape), np.uint8)
        else:
            self.single_observation_space = gym.spaces.Box(obs_space.low, obs_space.high, obs_space.shape, np.float32)
        self.single_action_space = gym.spaces.Discrete(self.env.num_actions)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.single_action_space.seed(seed)
        self.action_space.seed(seed)

        def reset_one(key: jax.Array) -> tuple[jax.Array, Any]:
            obs, state = self.env.reset(key, self.params)
            return self._observation(obs), state

        def step_one(key: jax.Array, state: Any, action: jax.Array) -> tuple:
            obs, state, reward, done, info = self.env.step(key, state, action, self.params)
            terminated = self.env.is_terminal(state, self.natural_params)
            truncated = done & ~terminated
            return self._observation(obs), state, reward, terminated, truncated, info

        self._reset_fn = jax.jit(jax.vmap(reset_one))

        def step_batch(key: jax.Array, states: Any, actions: jax.Array) -> tuple:
            step_key, reset_key = jax.random.split(key)
            obs, states, rewards, terminated, truncated, info = jax.vmap(step_one)(jax.random.split(step_key, num_envs), states, actions)
            done = terminated | truncated
            info = {name: value for name, value in info.items() if log_achievements or "achievement" not in name.lower()}
            info["episode_metrics"] = {name: value for name, value in info.items() if "achievement" in name.lower() or name == "score"}
            info["final_obs"] = obs
            info["_final_obs"] = done

            def reset_done() -> tuple[jax.Array, Any]:
                reset_obs, reset_states = self._reset_fn(jax.random.split(reset_key, self.num_reset_envs))
                # Assign distinct worlds first; reuse candidates if more slots finish.
                reset_indices = (jnp.cumsum(done) - 1) % self.num_reset_envs
                reset_obs = reset_obs[reset_indices]
                reset_states = jax.tree.map(lambda x: x[reset_indices], reset_states)
                states_out = jax.tree.map(lambda new, old: jnp.where(done.reshape((num_envs,) + (1,) * (old.ndim - 1)), new, old), reset_states, states)
                obs_out = jnp.where(done.reshape((num_envs,) + (1,) * (obs.ndim - 1)), reset_obs, obs)
                return obs_out, states_out

            # World generation runs only when at least one environment finishes.
            obs, states = jax.lax.cond(jnp.any(done), reset_done, lambda: (obs, states))
            return obs, states, rewards, terminated, truncated, info

        self._step_fn = jax.jit(step_batch)
        if render_mode == "rgb_array":
            if env_name.startswith("Craftax-Classic-"):
                from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
                block_pixel_size = 7
            else:
                from craftax.craftax.renderer import make_craftax_pixel_renderer
                block_pixel_size = 10
            self._render_fn = jax.jit(jax.vmap(make_craftax_pixel_renderer(block_pixel_size)))

    @partial(jax.jit, static_argnums=0)
    def _observation(self, obs: jax.Array) -> jax.Array:
        if self.pixel_obs:
            return jnp.clip(jnp.rint(obs * 255.0), 0, 255).astype(jnp.uint8)[None]
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
        self.rng, key = jax.random.split(self.rng)
        obs, self.states = self._reset_fn(jax.random.split(key, self.num_envs))
        return np.asarray(obs), {}

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        self.rng, key = jax.random.split(self.rng)
        actions = jnp.asarray(actions, dtype=jnp.int32).reshape(self.num_envs)
        obs, self.states, rewards, terminated, truncated, info = self._step_fn(key, self.states, actions)
        return jax.tree.map(np.asarray, (obs, rewards, terminated, truncated, info))

    def close(self, **kwargs: Any) -> None:
        self.states = None

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        return np.clip(np.asarray(self._render_fn(self.states)), 0, 255).astype(np.uint8)


def make_craftax_env(env_name: str, seed: int, num_envs: int, max_episode_steps: int | None = None, render_mode: str | None = None,
                     use_optimistic_resets: bool = False, optimistic_reset_ratio: int = 16, log_achievements: bool = False) -> CraftaxEnv:
    return CraftaxEnv(env_name, seed, num_envs, max_episode_steps=max_episode_steps, render_mode=render_mode,
                      use_optimistic_resets=use_optimistic_resets, optimistic_reset_ratio=optimistic_reset_ratio, log_achievements=log_achievements)
