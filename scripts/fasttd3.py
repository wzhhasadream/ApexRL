from dataclasses import dataclass
from typing import Literal

import tyro

from ApexRL.agents.fasttd3.config import FastTD3Config
from ApexRL.envs import create_envs
from ApexRL.runners import EnvironmentConfig, OffPolicyRunner, OffPolicyRunnerConfig


@dataclass
class Args(FastTD3Config):
    """Train FastTD3 with JAX or PyTorch."""

    backend: Literal["jax", "torch"] = "jax"
    env_type: Literal["mujoco", "dmc", "myosuite", "humanoid_bench", "metaworld", "playground", "isaaclab", "maniskill", "mjlab"] = "playground"
    env_id: str = "G1JoystickFlatTerrain"
    num_eval_envs: int = 10
    eval_episodes: int | None = None  # Defaults to one episode per evaluation environment.
    action_repeat: int | None = None
    max_episode_steps: int | None = None
    rescale_action: bool = True
    num_eval: int = 20
    num_log: int = 50
    results_dir: str = "Results"
    project: str = "ApexRL"
    run_name: str | None = None
    record_video: bool = False
    save_agent: bool = False
    save_onnx: bool = False


if __name__ == "__main__":
    args = tyro.cli(Args)
    cfg = FastTD3Config.from_cli(args, args.env_type, args.env_id)

    if args.backend == "jax":
        from ApexRL.agents.fasttd3.jax import FastTD3Agent
    else:
        from ApexRL.agents.fasttd3.torch import FastTD3Agent

    train_envs, eval_envs, record_envs = create_envs(
        env_name=args.env_id, env_type=args.env_type, seed=cfg.seed,
        num_train_envs=cfg.num_train_env, num_eval_envs=args.num_eval_envs, num_record_envs=1,
        rescale_action=args.rescale_action,
        action_repeat=args.action_repeat, max_episode_steps=args.max_episode_steps,
        clip_action=True, render_mode="rgb_array" if args.record_video else None,
    )
    eval_episodes = eval_envs.num_envs if args.eval_episodes is None else args.eval_episodes
    agent = FastTD3Agent(train_envs, cfg)
    env_cfg = EnvironmentConfig(env_type=args.env_type, seed=cfg.seed, eval_episode=eval_episodes)
    run_cfg = OffPolicyRunnerConfig(
        total_timesteps=cfg.total_timesteps,
        grad_step_per_interaction_step=cfg.grad_step_per_interaction_step,
        num_eval=args.num_eval, num_log=args.num_log,
        record_video=args.record_video, save_agent=args.save_agent, save_onnx=args.save_onnx,
    )
    runner = OffPolicyRunner(
        train_envs=train_envs, eval_envs=eval_envs, record_envs=record_envs, agent=agent,
        env_cfg=env_cfg, run_cfg=run_cfg, results_dir=args.results_dir, project=args.project,
        run_name=args.run_name or f"fasttd3-{args.backend}-{args.env_id}",
    )
    runner.run()
