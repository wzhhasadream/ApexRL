from dataclasses import dataclass
from typing import Literal

import tyro

from ApexRL.agents.pqn.config import PQNConfig
from ApexRL.agents.pqn.jax import PQNAgent
from ApexRL.envs import create_envs
from ApexRL.runners import OnPolicyRunner, OnPolicyRunnerConfig
from ApexRL.runners.types import EnvironmentConfig


@dataclass
class Args(PQNConfig):
    env_type: Literal["atari"] = "atari"   # PQN only support atari
    env_name: str = "Pong-v5"
    num_eval: int = 20
    num_log: int = 50
    eval_episodes: int = 50
    results_dir: str = "Results"
    project: str = "ApexRL"
    run_name: str | None = None
    record_video: bool = False
    save_agent: bool = False
    save_onnx: bool = False


args = tyro.cli(Args)

cfg = PQNConfig.from_cli(args, args.env_type, args.env_name)  # Defaults < environment presets < explicit CLI overrides. Omitted CLI arguments do not override presets.

train_envs, eval_envs, record_envs = create_envs(
    env_name=args.env_name,
    env_type=args.env_type,
    seed=cfg.seed,
    num_train_envs=cfg.num_train_env,
    num_eval_envs=50,   # NOTE: eval_episodes must be a multiple of num_eval_envs.
    num_record_envs=1,
    action_repeat=4,
    max_episode_steps=108_000,
    clip_action=False,
    render_mode="rgb_array" if args.record_video else None,
)

agent = PQNAgent(train_envs, cfg)
env_cfg = EnvironmentConfig(env_type=args.env_type, seed=cfg.seed, eval_episode=args.eval_episodes)
run_cfg = OnPolicyRunnerConfig(
    total_timesteps=cfg.total_timesteps,
    rollout_steps=cfg.rollout_steps,
    gamma=cfg.gamma,
    num_eval=args.num_eval,
    num_log=args.num_log,
    record_video=args.record_video,
    save_agent=args.save_agent,
    save_onnx=args.save_onnx,
)
runner = OnPolicyRunner(
    train_envs=train_envs,
    eval_envs=eval_envs,
    record_envs=record_envs,
    agent=agent,
    env_cfg=env_cfg,
    run_cfg=run_cfg,
    results_dir=args.results_dir,
    project=args.project,
    run_name=args.run_name or args.env_name,
)
runner.run()
