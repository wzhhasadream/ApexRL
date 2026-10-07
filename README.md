# ApexRL

**Fast, state-of-the-art reinforcement learning algorithms in PyTorch and JAX.**

ApexRL collects recent, high-performing RL algorithms behind one interface. Each
algorithm has a PyTorch implementation and a JAX/Flax NNX implementation. Both
share the same config, rollout/replay buffers, environment wrappers, and
training runners. The focus is wall-clock speed on a single GPU:

- **PyTorch**: `torch.compile` (`max-autotune`), fused optimizers, and pinned-memory host-to-device copies.
- **JAX**: fully `jit`-compiled updates with `lax.scan` over minibatches, and parameters kept on the device.
- **Shared**: buffers that live on the GPU, uint8 image observations, and envpool-backed Atari (C++ thread pool, no per-env subprocesses).

## Algorithms

| Algorithm | Type | Action space | PyTorch | JAX | Reference |
|---|---|---|:-:|:-:|---|
| **PPO** | On-policy actor-critic (clipped / SPO objective, GAE, KL-adaptive lr) | Continuous & discrete (incl. Atari pixels) | ✓ | ✓ | [Schulman et al., 2017](https://arxiv.org/abs/1707.06347); SPO objective: [Xie et al., ICML 2025](https://arxiv.org/abs/2401.16025) |
| **PQN** | On-policy Q-learning (λ-returns, no replay/target net) | Discrete | ✓ | ✓ | [Gallici et al., ICLR 2025](https://arxiv.org/abs/2407.04811) |
| **FastTD3** | Off-policy actor-critic (distributional TD3, massively parallel envs) | Continuous | ✓ | ✓ | [Seo et al., 2025](https://arxiv.org/abs/2505.22642) |
| **FastSAC** | Off-policy actor-critic (SAC tuned for massively parallel envs) | Continuous | ✓ | ✓ | [Seo et al., 2025](https://arxiv.org/abs/2512.01996) |
| BQN | Off-policy value-based | Discrete | ✓ | ✓ | |
| **V-Simba** | Off-policy visual actor-critic (ConvNeXt-style encoder + Simba MLPs, categorical critic) | Continuous (pixels) | ✓ | ✓ | [Kim et al., RLJ 2026](https://github.com/DAVIAN-Robotics/V-Simba) |
| RePT | Off-policy visual actor-critic | Continuous | ✓ | ✓ | |

## Environments

`ApexRL.envs.create_envs(env_name, env_type, seed, ...)` returns `(train_envs, eval_envs, record_envs)`
as gymnasium `VectorEnv`s. Supported `env_type` values:

- **CPU simulators**: `atari`, `mujoco`, `dmc`, `myosuite`, `humanoid_bench`, `metaworld`
- **GPU simulators**: `playground`, `isaaclab`, `maniskill`, `mjlab`

Atari training and evaluation run on [envpool](https://github.com/sail-sg/envpool), with the same
settings as purejaxql: life-loss episodes and reward clipping during training only, no sticky
actions, up to 30 no-ops on reset, and FIRE on reset. envpool cannot render, so video recording
uses a single gymnasium ALE env configured the same way.

## Installation

Requires Python ≥ 3.11 and an NVIDIA GPU.

```bash
pip install torch jax[cuda12] flax optax orbax-checkpoint gymnasium ale-py envpool
pip install -e .
```

## Quick start: PQN on Atari

```python
from ApexRL.agents.pqn.config import PQNConfig
from ApexRL.agents.pqn.torch import PQNAgent  # or: from ApexRL.agents.pqn.jax import PQNAgent
from ApexRL.envs import create_envs
from ApexRL.runners import EnvironmentConfig, OnPolicyRunner, OnPolicyRunnerConfig

cfg = PQNConfig(seed=1)
train_envs, eval_envs, record_envs = create_envs(
    "Breakout-v5", "atari", cfg.seed, num_train_envs=cfg.num_envs, num_eval_envs=8, clip_action=False
)
agent = PQNAgent(train_envs, cfg)
runner = OnPolicyRunner(
    train_envs=train_envs, eval_envs=eval_envs, record_envs=record_envs, agent=agent,
    env_cfg=EnvironmentConfig(env_type="atari", seed=cfg.seed, eval_episode=8),
    run_cfg=OnPolicyRunnerConfig(total_timesteps=cfg.total_timesteps, rollout_steps=cfg.rollout_steps, gamma=cfg.gamma),
    results_dir="Results", project="ApexRL", run_name="pqn-breakout",
)
runner.run()
```

## Quick start: PPO

PPO uses the same `OnPolicyRunner`. The default `PPOConfig()` targets massively parallel continuous
control (rsl_rl-style: KL-adaptive lr, RMS-normalized observations, MLP). `PPOConfig.atari()` gives the
CleanRL Atari setting: a Nature CNN on stacked uint8 frames, categorical actions, and a linearly annealed lr.

```python
from ApexRL.agents.ppo.config import PPOConfig
from ApexRL.agents.ppo.torch import PPOAgent  # or: from ApexRL.agents.ppo.jax import PPOAgent

cfg = PPOConfig.atari(seed=1)
train_envs, eval_envs, record_envs = create_envs(
    "Breakout-v5", "atari", cfg.seed, num_train_envs=cfg.num_envs, num_eval_envs=8, clip_action=False
)
agent = PPOAgent(train_envs, cfg)
# ... then OnPolicyRunner exactly as above, with rollout_steps=cfg.rollout_steps
```

Continuous actions are sampled from an unbounded Gaussian, so PPO accepts environments with infinite action
bounds. Off-policy agents squash actions with tanh: they use finite env bounds as-is and fall back to [-1, 1]
(with a warning) for unbounded dimensions.

The off-policy algorithms (FastTD3, FastSAC, BQN, ...) use `OffPolicyRunner` and `OffPolicyRunnerConfig`
in the same way. See `main.py` for a complete BQN script.

## Project layout

```
ApexRL/
├── agents/     # one folder per algorithm: config.py + torch/ + jax/
├── buffers/    # on-policy rollout buffers and off-policy replay buffers (torch & jax)
├── envs/       # create_envs() and per-simulator wrappers
├── model/      # shared layers, ensembles, policies, normalizers, backbones (e.g. NatureCNN) (torch & jax)
├── runners/    # OnPolicyRunner / OffPolicyRunner training loops
└── common/     # evaluation, logging, device helpers
```

## License

ApexRL is released under the [MIT License](LICENSE).

Some algorithms are adapted from third-party code and keep their original licenses. Each adapted
file names its upstream in a header, and the full license texts are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

| Component | Upstream | License |
|---|---|---|
| `ApexRL/agents/pqn/` | [mttga/purejaxql](https://github.com/mttga/purejaxql) | Apache-2.0 |
| `ApexRL/agents/fasttd3/` | [younggyoseo/FastTD3](https://github.com/younggyoseo/FastTD3) | MIT |
| `ApexRL/agents/fastsac/` | [amazon-far/holosoma](https://github.com/amazon-far/holosoma) | Apache-2.0 |
| `ApexRL/agents/ppo/` | [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl), [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) | MIT, BSD-3-Clause |
| `ApexRL/agents/v_simba/` | [DAVIAN-Robotics/V-Simba](https://github.com/DAVIAN-Robotics/V-Simba) | Apache-2.0 |

## Citation

If you use these algorithms, please cite the original papers:

```bibtex
@article{schulman2017ppo,
  title   = {Proximal Policy Optimization Algorithms},
  author  = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal = {arXiv preprint arXiv:1707.06347},
  year    = {2017}
}

@inproceedings{xie2025simple,
  title     = {Simple Policy Optimization},
  author    = {Xie, Zhengpeng and Zhang, Qiang and Yang, Fan and Hutter, Marco and Xu, Renjing},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=SG8Yx1FyeU}
}

@inproceedings{gallici2025simplifying,
  title     = {Simplifying Deep Temporal Difference Learning},
  author    = {Gallici, Matteo and Fellows, Mattie and Ellis, Benjamin and Pou, Bartomeu and Masmitja, Ivan and Foerster, Jakob Nicolaus and Martin, Mario},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2407.04811}
}

@article{seo2025fasttd3,
  title   = {FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control},
  author  = {Seo, Younggyo and Sferrazza, Carmelo and Geng, Haoran and Nauman, Michal and Yin, Zhao-Heng and Abbeel, Pieter},
  journal = {arXiv preprint arXiv:2505.22642},
  year    = {2025}
}

@article{seo2025fastsac,
  title   = {Learning Sim-to-Real Humanoid Locomotion in 15 Minutes},
  author  = {Seo, Younggyo and Sferrazza, Carmelo and Chen, Juyue and Shi, Guanya and Duan, Rocky and Abbeel, Pieter},
  journal = {arXiv preprint arXiv:2512.01996},
  year    = {2025}
}

@article{kim2026vsimba,
  title   = {V-Simba: Unleashing the Architectural Potential of RL in Visual Continuous Control},
  author  = {Kim, Donghu and Lee, Youngdo and Lee, Hojoon and Obando-Ceron, Johan and Lee, Byungkun and Courville, Aaron and Castro, Pablo Samuel and Choo, Jaegul and Lyle, Clare},
  journal = {Reinforcement Learning Journal},
  year    = {2026}
}
```
