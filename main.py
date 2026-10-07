import argparse

from ApexRL.agents.pqn.config import PQNConfig
from ApexRL.agents.pqn.jax import PQNAgent
from ApexRL.envs import create_envs
from ApexRL.runners import OnPolicyRunner, OnPolicyRunnerConfig
from ApexRL.runners.types import EnvironmentConfig


parser = argparse.ArgumentParser(description="Train PQN on Atari.")
parser.add_argument("--env", default="Pong-v5", help="Atari environment name, for example Pong or ALE/Pong-v5.")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--total-timesteps", type=int, default=None)
parser.add_argument("--rollout-steps", type=int, default=None)
parser.add_argument("--num-eval", type=int, default=20)
parser.add_argument("--num-log", type=int, default=50)
parser.add_argument("--eval-episodes", type=int, default=50)
parser.add_argument("--results-dir", default="Results")
parser.add_argument("--project", default="ApexRL")
parser.add_argument("--run-name", default=None)
parser.add_argument("--record-video", action="store_true")
parser.add_argument("--save-agent", action="store_true")
parser.add_argument("--save-onnx", action="store_true")
args = parser.parse_args()

cfg = PQNConfig.from_env("atari", args.env, seed=args.seed, num_envs=args.num_envs)
if args.total_timesteps is not None:
    cfg.total_timesteps = args.total_timesteps
if args.rollout_steps is not None:
    cfg.rollout_steps = args.rollout_steps

train_envs, eval_envs, record_envs = create_envs(
    env_name=args.env,
    env_type="atari",
    seed=args.seed,
    num_train_envs=cfg.num_envs,
    num_eval_envs=50,
    num_record_envs=1,
    action_repeat=4,
    max_episode_steps=108_000,
    clip_action=False,
    render_mode="rgb_array" if args.record_video else None,
)

agent = PQNAgent(train_envs, cfg)
env_cfg = EnvironmentConfig(env_type="atari", seed=args.seed, eval_episode=args.eval_episodes)
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
    run_name=args.run_name or args.env,
)
runner.run()
