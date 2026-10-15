from dataclasses import dataclass
from typing import Literal

import tyro

from ApexRL.agents.rept.config import RePTConfig
from ApexRL.envs import create_envs
from ApexRL.runners import EnvironmentConfig, OffPolicyRunner, OffPolicyRunnerConfig


@dataclass
class Args(RePTConfig):
    """Train RePT with JAX."""

    backend: Literal["jax"] = "jax"
    env_type: Literal["mujoco", "dmc", "myosuite", "humanoid_bench", "metaworld"] = "mujoco"
    env_id: str = "HalfCheetah-v4"
    num_eval_envs: int = 1
    eval_episodes: int | None = 10  # None selects one episode per evaluation environment.
    max_episode_steps: int | None = None
    rescale_action: bool = True
    num_eval: int = 20
    num_log: int = 50
    results_dir: str = "RePT-Result"
    record_video: bool = False
    save_agent: bool = False
    save_onnx: bool = False
    encoder_path: str | None = None
    save_encoder: bool = False


if __name__ == "__main__":
    args = tyro.cli(Args)
    cfg = RePTConfig.from_cli(args, args.env_type, args.env_id)

    from ApexRL.agents.rept.jax import RePTAgent

    train_envs, eval_envs, record_envs = create_envs(
        env_name=args.env_id, env_type=args.env_type, seed=cfg.seed,
        num_train_envs=cfg.num_train_env, num_eval_envs=args.num_eval_envs, num_record_envs=1,
        rescale_action=args.rescale_action,
        action_repeat=cfg.action_repeat, max_episode_steps=args.max_episode_steps,
        clip_action=True, render_mode="rgb_array" if args.record_video else None,
    )
    args.eval_episodes = eval_envs.num_envs if args.eval_episodes is None else args.eval_episodes
    agent = RePTAgent(train_envs, cfg)
    if args.encoder_path is not None:
        agent.load_encoder(f"args.encoder_path/{args.seed}")
    env_cfg = EnvironmentConfig(env_type=args.env_type, seed=cfg.seed, eval_episode=args.eval_episodes)
    run_cfg = OffPolicyRunnerConfig(
        total_timesteps=cfg.total_timesteps,
        grad_step_per_interaction_step=cfg.grad_step_per_interaction_step,
        num_eval=args.num_eval, num_log=args.num_log,
        record_video=args.record_video, save_agent=args.save_agent, save_onnx=args.save_onnx,
    )
    runner = OffPolicyRunner(
        train_envs=train_envs, eval_envs=eval_envs, record_envs=record_envs, agent=agent,
        env_cfg=env_cfg, run_cfg=run_cfg, results_dir=args.results_dir, project=args.env_type,
        run_name=args.env_id,
    )
    save_path = runner.log_dir
    runner.run()
    if args.save_encoder:
        agent.save_encoder(save_path)
