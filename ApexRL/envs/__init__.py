from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv, VectorEnv, AsyncVectorEnv
from gymnasium.wrappers import AddRenderObservation, FrameStackObservation, RescaleAction, TimeLimit

from .wrapper import ActionRepeat, wrap_vector_env

if TYPE_CHECKING:
    from .dmc import DMCPhysicsMods

CPU_SIM = ("mujoco", "dmc", "myosuite", "humanoid_bench", "metaworld")
GPU_SIM = ("playground", "isaaclab", "maniskill", "mjlab")


def create_envs(
    env_name: str,
    env_type: str,
    seed: int,
    num_train_envs: int = 1,
    num_eval_envs: int = 1,
    num_record_envs: int = 1,
    rescale_action: bool = True,
    action_repeat: int | None = None,
    max_episode_steps: int | None = None,
    clip_action: bool = True,
    render_mode: str | None = None,
    physics_mods: DMCPhysicsMods | None = None,
) -> tuple[VectorEnv, VectorEnv, VectorEnv]:
    if action_repeat is None:
        action_repeat = 4 if env_type == "atari" else 1
    if max_episode_steps is None:
        max_episode_steps = 108_000 if env_type == "atari" else 1000

    # IsaacLab uses one simulator shared by training, evaluation, and recording.
    if env_type == "isaaclab":
        from .isaaclab import make_isaaclab_env

        env = make_isaaclab_env(env_name=env_name, seed=seed, num_envs=num_train_envs, headless=True, render_mode=render_mode)
        env = wrap_vector_env(env, action_repeat=action_repeat, clip_action=clip_action)
        return env, env, env

    envs = []
    for num_envs, mode, training, record in ((num_train_envs, None, True, False), (num_eval_envs, None, False, False), (num_record_envs, render_mode, False, True)):
        vector_repeat = action_repeat
        if env_type == "metaworld" and env_name.upper() in ("MT10", "MT50"):
            from .metaworld import make_metaworld_benchmark_envs

            env = make_metaworld_benchmark_envs(
                benchmark_name=env_name, seed=seed, num_envs=num_envs, render_mode=mode,
                max_episode_steps=max_episode_steps, use_one_hot=True,
            )
        elif env_type == "atari" and not record:
            from .atari import make_envpool_atari_env

            env = make_envpool_atari_env(env_name, num_envs, seed, training=training, action_repeat=action_repeat, max_episode_steps=max_episode_steps)
            vector_repeat = 1
        elif env_type == "atari":
            # envpool cannot render, so the record env is a gymnasium env with matching settings.
            from .atari import make_atari_env

            env = SyncVectorEnv([lambda i=i: make_atari_env(env_name, seed + i, render_mode=mode, action_repeat=action_repeat, max_episode_steps=max_episode_steps) for i in range(num_envs)], autoreset_mode="SameStep")
            vector_repeat = 1
        elif env_type in CPU_SIM:
            env = create_vec_env(
                env_type=env_type, env_name=env_name, num_envs=num_envs, seed=seed,
                rescale_action=rescale_action, max_episode_steps=max_episode_steps,
                render_mode=mode, action_repeat=action_repeat, physics_mods=physics_mods,
            )
            vector_repeat = 1
        elif env_type == "playground":
            from .playground import make_playground_env

            env = make_playground_env(env_name=env_name, seed=seed, num_envs=num_envs, max_episode_steps=max_episode_steps, action_repeat=action_repeat)
            vector_repeat = 1
        elif env_type == "maniskill":
            from .maniskill import make_maniskill_env

            env = make_maniskill_env(env_name=env_name, num_envs=num_envs, render_mode=mode)
        elif env_type == "mjlab":
            from .mjlab import make_mjlab_env

            env = make_mjlab_env(task_id=env_name, seed=seed, num_envs=num_envs, render_mode=mode)
        else:
            raise ValueError(f"Unsupported env_type: {env_type}")

        env = wrap_vector_env(env, action_repeat=vector_repeat, clip_action=clip_action and env_type != "atari")
        envs.append(env)

    return envs[0], envs[1], envs[2]


def create_vec_env(
    env_type: str,
    env_name: str,
    num_envs: int,
    seed: int,
    rescale_action: bool = True,
    max_episode_steps: int | None = None,
    render_mode: str | None = None,
    action_repeat: int | None = None,
    physics_mods: DMCPhysicsMods | None = None,
) -> VectorEnv:
    if action_repeat is None:
        action_repeat = 1
    if max_episode_steps is None:
        max_episode_steps = 1000
    is_pixel_obs = env_name.endswith("-visual")
    env_name = env_name.removesuffix("-visual")
    if is_pixel_obs:
        render_mode = "rgb_array"

    def make_one_env(index: int) -> gym.Env:
        env_seed = seed + index
        if env_type == "dmc":
            from .dmc import make_dmc_env

            env = make_dmc_env(env_name, env_seed, render_mode=render_mode, physics_mods=physics_mods)
        elif env_type == "mujoco":
            from .mujoco import make_mujoco_env

            env = make_mujoco_env(env_name, env_seed, render_mode=render_mode)
        elif env_type == "humanoid_bench":
            from .humanoid_bench import make_humanoid_env

            env = make_humanoid_env(env_name, env_seed, render_mode=render_mode)
        elif env_type == "myosuite":
            from .myosuite import make_myosuite_env

            env = make_myosuite_env(env_name, env_seed, render_mode=render_mode)
        elif env_type == "metaworld":
            from .metaworld import make_metaworld_env

            env = make_metaworld_env(env_name, env_seed, render_mode=render_mode)
        else:
            raise ValueError(f"Unsupported CPU env_type: {env_type}")

        if rescale_action and isinstance(env.action_space, gym.spaces.Box):
            env = RescaleAction(env, np.float32(-1.0), np.float32(1.0))
        # Count raw environment steps before action repeat and frame stacking.
        env = TimeLimit(env, max_episode_steps)
        if action_repeat > 1:
            env = ActionRepeat(env, action_repeat)
        if is_pixel_obs:
            env = AddRenderObservation(env, render_only=True)
            env = FrameStackObservation(env, stack_size=3)

        env.observation_space.seed(env_seed)
        env.action_space.seed(env_seed)
        return env

    env_fns = [lambda i=i: make_one_env(i) for i in range(num_envs)]
    if num_envs == 1:
        return SyncVectorEnv(env_fns, autoreset_mode="SameStep")
    return AsyncVectorEnv(env_fns, autoreset_mode="SameStep", context="spawn")
