import argparse

from ApexRL.agents.bqn.config import BQNConfig
from ApexRL.agents.bqn.jax import BQNAgent
from ApexRL.envs import create_envs
from ApexRL.runners import OffPolicyRunner, OffPolicyRunnerConfig
from ApexRL.runners.types import EnvironmentConfig


parser = argparse.ArgumentParser(description="Train BQN on Atari.")
parser.add_argument("--env", default="Pong-v5", help="Atari environment name, for example Pong or ALE/Pong-v5.")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--total-timesteps", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=None)
parser.add_argument("--learning-starts", type=int, default=None)
parser.add_argument("--use-bn", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--normalize-rewards", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--num-eval", type=int, default=20)
parser.add_argument("--num-log", type=int, default=50)
parser.add_argument("--eval-episodes", type=int, default=10)
parser.add_argument("--results-dir", default="Results")
parser.add_argument("--project", default="ApexRL")
parser.add_argument("--run-name", default=None)
parser.add_argument("--record-video", action="store_true")
parser.add_argument("--save-agent", action="store_true")
parser.add_argument("--save-onnx", action="store_true")
args = parser.parse_args()

cfg = BQNConfig(seed=args.seed)
if args.total_timesteps is not None:
    cfg.total_timesteps = args.total_timesteps
if args.batch_size is not None:
    cfg.batch_size = args.batch_size
if args.learning_starts is not None:
    cfg.learning_starts = args.learning_starts
if args.use_bn is not None:
    cfg.use_bn = args.use_bn
if args.normalize_rewards is not None:
    cfg.normalize_rewards = args.normalize_rewards

train_envs, eval_envs, record_envs = create_envs(
    env_name=args.env,
    env_type="atari",
    seed=args.seed,
    num_train_envs=args.num_envs,
    num_eval_envs=1,
    num_record_envs=1,
    action_repeat=4,
    max_episode_steps=108_000,
    clip_action=False,
    render_mode="rgb_array" if args.record_video else None,
)

agent = BQNAgent(train_envs, cfg)
env_cfg = EnvironmentConfig(env_type="atari", seed=args.seed, eval_episode=args.eval_episodes)
run_cfg = OffPolicyRunnerConfig(
    total_timesteps=cfg.total_timesteps,
    grad_step_per_interaction_step=cfg.grad_step_per_interaction_step,
    num_eval=args.num_eval,
    num_log=args.num_log,
    record_video=args.record_video,
    save_agent=args.save_agent,
    save_onnx=args.save_onnx,
)
runner = OffPolicyRunner(
    train_envs=train_envs,
    eval_envs=eval_envs,
    record_envs=record_envs,
    agent=agent,
    env_cfg=env_cfg,
    run_cfg=run_cfg,
    results_dir=args.results_dir,
    project=args.project,
    run_name=args.env,
)
runner.run()
