import dataclasses

import optuna
import tyro

from ApexRL.agents.bqn.config import BQNConfig
from ApexRL.agents.bqn.jax import BQNAgent
from ApexRL.common import tune
from ApexRL.envs import create_envs
from ApexRL.runners import EnvironmentConfig, OffPolicyRunner, OffPolicyRunnerConfig

@dataclasses.dataclass
class Args:
    env_type: str = "atari"
    env_id: str = "Pong-v5"
    seed: int = 1
    total_timesteps: int = int(5e7)
    time_out: float = 7200
    eval_count: int = 20
    n_trials: int = 20
    study_name: str = "bqn-pong"
    storage: str | None = "sqlite:///bqn_tuning.db"


args = tyro.cli(Args)


def sample_bqn_params(trial: optuna.Trial) -> dict:
    params = {
        "num_train_env": trial.suggest_categorical(
            "num_train_env", [128, 256, 512]
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", [32, 64, 128, 256]
        ),
        "buffer_size": trial.suggest_categorical(
            "buffer_size", [int(1e6), int(2e6), int(4e6), int(5e6)]
        ),
        "grad_step_per_interaction_step": trial.suggest_categorical(
            "grad_step_per_interaction_step", [0.5, 1, 2, 4]
        ),
    }
    return params


def objective(trial: optuna.Trial):
    params = sample_bqn_params(trial)
    cfg = BQNConfig.from_env(
        args.env_type,
        args.env_id,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        **params,
    )

    train_envs, eval_envs, record_envs = create_envs(
        env_name=args.env_id,
        env_type=args.env_type,
        seed=cfg.seed,
        num_train_envs=cfg.num_train_env,
        num_eval_envs=50,
        num_record_envs=1,
        clip_action=False,
    )

    agent = BQNAgent(train_envs, cfg)

    env_cfg = EnvironmentConfig(
        env_type=args.env_type,
        seed=cfg.seed,
        eval_episode=eval_envs.num_envs,
    )

    run_cfg = OffPolicyRunnerConfig(
        total_timesteps=cfg.total_timesteps,
        grad_step_per_interaction_step=cfg.grad_step_per_interaction_step,
        num_eval=args.eval_count,
        num_log=50,
    )

    runner = OffPolicyRunner(
        train_envs=train_envs,
        eval_envs=eval_envs,
        record_envs=record_envs,
        agent=agent,
        env_cfg=env_cfg,
        run_cfg=run_cfg,
        results_dir="Results",
        project="ApexRL-Tune",
        run_name=f"bqn-trial-{trial.number}",
        max_time=args.time_out
    )
    return runner.run()


study = tune(
    objective,
    n_trials=args.n_trials,
    sampler=optuna.samplers.TPESampler(n_startup_trials=5, seed=args.seed),
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    study_name=args.study_name,
    storage=args.storage,
)

print(study.best_params)
print(study.best_value)
