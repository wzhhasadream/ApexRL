

![ApexRL](assets/logo-light.svg)

**Fast, state-of-the-art reinforcement learning in PyTorch and JAX — one interface, two backends.**

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-torch.compile-EE4C2C?logo=pytorch&logoColor=white)
![JAX](https://img.shields.io/badge/JAX-Flax%20NNX-A259FF)
![License: MIT](https://img.shields.io/badge/license-MIT-22C55E)

[Algorithms](#algorithms) •
[Installation](#installation) •
[Quick start](#quick-start) •
[Environment presets](#environment-presets) •
[Environments](#environments) •
[Citation](#citation)



---

ApexRL collects recent, high-performing RL algorithms behind a single interface. Every algorithm ships
with a **PyTorch** and a **JAX/Flax NNX** implementation that share the same config, buffers, environment
wrappers, and training runners, so you can switch backends by changing one import.

## Highlights

- **Built for wall-clock speed on one GPU.** PyTorch paths use `torch.compile(mode="max-autotune")` with CUDA graphs,
  fused optimizers, and TF32. JAX paths `jit` the whole update and `lax.scan` over minibatches.
- **Data stays on the GPU.** Rollout and replay buffers live on device, observations are uploaded once per step,
  and image observations are stored as `uint8`.
- **Massively parallel environments.** GPU simulators (MuJoCo Playground, IsaacLab, ManiSkill, mjlab) and
  envpool-backed Atari.
- **Faithful to the original papers.** Defaults follow the upstream implementations, and each adapted file names its source.

## Algorithms


| Algorithm   | Type                                                                                        | Action space                               | PyTorch | JAX | Reference                                                                                                                 |
| ----------- | ------------------------------------------------------------------------------------------- | ------------------------------------------ | ------- | --- | ------------------------------------------------------------------------------------------------------------------------- |
| **PPO**     | On-policy actor-critic (clipped / SPO objective, GAE, KL-adaptive lr)                       | Continuous & discrete (incl. Atari pixels) | ✓       | ✓   | [Schulman et al., 2017](https://arxiv.org/abs/1707.06347); SPO: [Xie et al., ICML 2025](https://arxiv.org/abs/2401.16025) |
| **PQN**     | Online Q-learning (λ-returns, no replay / target net)                                       | Discrete                                   | ✓       | ✓   | [Gallici et al., ICLR 2025](https://arxiv.org/abs/2407.04811)                                                             |
| **FastTD3** | Off-policy actor-critic (distributional TD3, massively parallel envs)                       | Continuous                                 | ✓       | ✓   | [Seo et al., 2025](https://arxiv.org/abs/2505.22642)                                                                      |
| **FastSAC** | Off-policy actor-critic (SAC tuned for massively parallel envs)                             | Continuous                                 | ✓       | ✓   | [Seo et al., 2025](https://arxiv.org/abs/2512.01996)                                                                      |
| **WarpSAC** | Off-policy actor-critic (replay decay, repeated exploration actions, distributional critic) | Continuous                                 | ✓       | ✓   | [Wu et al., 2026](https://arxiv.org/abs/2608.24479); [original implementation](https://github.com/wzhhasadream/warprl)    |
| **V-Simba** | Off-policy visual actor-critic (ConvNeXt-style encoder + Simba MLPs, categorical critic)    | Continuous (pixels)                        | ✓       | ✓   | [Kim et al., RLJ 2026](https://github.com/DAVIAN-Robotics/V-Simba)                                                        |

PQN uses `OnPolicyRunner` in ApexRL: collect a fresh rollout, update the Q-network with λ-returns,
then clear the rollout buffer. It has no experience replay or target network. Its Q-learning targets
are off-policy; the runner groups algorithms by their data collection and update loop.



## Installation

ApexRL requires Python 3.11 and an NVIDIA GPU. `pyproject.toml` pins PyTorch 2.9 (CUDA 12.8)
and JAX 0.6 (CUDA 12).

Install with [uv](https://github.com/astral-sh/uv) into a conda env and pick the extras you need:

```bash
conda create -n apexrl python=3.11 -y && conda activate apexrl
pip install uv

uv pip install -e ".[torch,jax,atari,mujoco]"   # e.g. Atari + MuJoCo with both backends
```

**Full install**: every simulator, including Isaac Lab and ManiSkill

  


```bash
export OMNI_KIT_ACCEPT_EULA=YES   # accept the Isaac Sim EULA non-interactively
uv pip install -e ".[all]"
```




| Extra                                                      | Installs                                        | Needed for               |
| ---------------------------------------------------------- | ----------------------------------------------- | ------------------------ |
| `all`                                                      | Both backends and every environment extra below | full installation        |
| `torch` / `jax`                                            | PyTorch 2.9 / JAX 0.6 + Flax NNX, Optax, Orbax  | the PyTorch / JAX agents |
| `atari`                                                    | ALE, envpool                                    | `env_type="atari"`       |
| `mujoco`, `dmc`, `myosuite`, `humanoid-bench`, `metaworld` | CPU MuJoCo-based suites                         | the matching `env_type`  |
| `playground`, `mjlab`                                      | MuJoCo Playground (MJX / Warp), mjlab           | GPU MuJoCo simulators    |
| `isaaclab`                                                 | Isaac Sim 5.1, Isaac Lab 2.3                    | `env_type="isaaclab"`    |


Use `uv pip` for extras. Isaac Sim, MetaWorld, Isaac Lab, and mjlab specify conflicting dependency versions;
the choices in `[tool.uv] override-dependencies` apply only when installing with uv.

## Quick start

PPO and PQN use `OnPolicyRunner`. FastTD3, FastSAC, WarpSAC, and V-Simba use `OffPolicyRunner`.

Create the config, environments, and agent, then pass them to a runner. This PPO example uses the Atari preset,
with one shared Nature CNN for actor and critic and gradient clipping over the complete actor-critic model:

```python
from ApexRL.agents.ppo.config import PPOConfig
from ApexRL.agents.ppo.jax import PPOAgent
from ApexRL.envs import create_envs
from ApexRL.runners import EnvironmentConfig, OnPolicyRunner, OnPolicyRunnerConfig

cfg = PPOConfig.from_env("atari", "Breakout-v5", seed=1)
train_envs, eval_envs, record_envs = create_envs(
    "Breakout-v5", "atari", cfg.seed, num_train_envs=cfg.num_train_env, num_eval_envs=8, clip_action=False
)
agent = PPOAgent(train_envs, cfg)
runner = OnPolicyRunner(
    train_envs=train_envs, eval_envs=eval_envs, record_envs=record_envs, agent=agent,
    env_cfg=EnvironmentConfig(env_type="atari", seed=cfg.seed, eval_episode=8),
    run_cfg=OnPolicyRunnerConfig(total_timesteps=cfg.total_timesteps, rollout_steps=cfg.rollout_steps, gamma=cfg.gamma),
    results_dir="Results", project="ApexRL", run_name="ppo-breakout",
)
runner.run()
```

All agent configs expose `num_train_env`. Pass it to
`create_envs(num_train_envs=...)` to set the training environment count, as in this WarpSAC example:

```python
from ApexRL.agents.warpsac.config import WarpSACConfig
from ApexRL.agents.warpsac.jax import WarpSACAgent  # or: from ApexRL.agents.warpsac.torch import WarpSACAgent
from ApexRL.envs import create_envs
from ApexRL.runners import EnvironmentConfig, OffPolicyRunner, OffPolicyRunnerConfig

cfg = WarpSACConfig.from_env("mujoco", "HalfCheetah-v5", seed=1)
train_envs, eval_envs, record_envs = create_envs("HalfCheetah-v5", "mujoco", cfg.seed, num_train_envs=cfg.num_train_env)
agent = WarpSACAgent(train_envs, cfg)
runner = OffPolicyRunner(
    train_envs=train_envs, eval_envs=eval_envs, record_envs=record_envs, agent=agent,
    env_cfg=EnvironmentConfig(env_type="mujoco", seed=cfg.seed, eval_episode=10),
    run_cfg=OffPolicyRunnerConfig(total_timesteps=cfg.total_timesteps, grad_step_per_interaction_step=cfg.grad_step_per_interaction_step),
    results_dir="Results", project="ApexRL", run_name="warpsac-halfcheetah",
)
runner.run()
```

PPO accepts unbounded continuous actions. Off-policy agents use finite action bounds and fall back to
`[-1, 1]` for unbounded dimensions.

V-Simba trains on rendered observations from CPU simulators. It uses `OffPolicyRunner` with bounded
continuous actions. The actor maps its output to each dimension's action bounds;
use `rescale_action=False` to keep the environment's native action space:

Install the `jax` and `dmc` extras for this example. On headless machines, set `MUJOCO_GL=egl` before launching Python.

```python
from ApexRL.agents.v_simba.config import VSimbaConfig
from ApexRL.agents.v_simba.jax import VSimbaAgent  # or: from ApexRL.agents.v_simba.torch import VSimbaAgent
from ApexRL.envs import create_envs
from ApexRL.runners import EnvironmentConfig, OffPolicyRunner, OffPolicyRunnerConfig

cfg = VSimbaConfig.from_env("dmc", "cheetah-run-visual", seed=1)
train_envs, eval_envs, record_envs = create_envs(
    "cheetah-run-visual", "dmc", cfg.seed, num_train_envs=cfg.num_train_env,
    action_repeat=cfg.action_repeat, max_episode_steps=cfg.max_episode_steps,
)
agent = VSimbaAgent(train_envs, cfg)
runner = OffPolicyRunner(
    train_envs=train_envs, eval_envs=eval_envs, record_envs=record_envs, agent=agent,
    env_cfg=EnvironmentConfig(env_type="dmc", seed=cfg.seed, eval_episode=10),
    run_cfg=OffPolicyRunnerConfig(total_timesteps=cfg.total_timesteps, grad_step_per_interaction_step=cfg.grad_step_per_interaction_step),
    results_dir="Results", project="ApexRL", run_name="vsimba-cheetah-run",
)
runner.run()
```

The defaults collect 500,000 transitions with `action_repeat=2`, corresponding to up to 1,000,000 simulator
steps. Pass `cfg.action_repeat` and `cfg.max_episode_steps` to `create_envs` to match the config.

## Environment presets

All six config classes support `Config.from_env(env_type, env_name=None, **overrides)`. It applies the
environment family, then the specific `env_type`, then explicit overrides. The current configs have no
`env_name`-specific presets.


| Config          | Example                                          | Preset applied                                                                                 |
| --------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `PPOConfig`     | `PPOConfig.from_env("atari", "Breakout-v5")`     | Atari rollout, optimizer, and network settings; CPU and GPU simulator presets also exist       |
| `PQNConfig`     | `PQNConfig.from_env("atari", "Breakout-v5")`     | Atari defaults from the config class                                                           |
| `FastTD3Config` | `FastTD3Config.from_env("playground")`           | 1,024 environments, `gamma=0.97`, critic value range `[-10, 10]`; Isaac Lab has its own preset |
| `FastSACConfig` | `FastSACConfig.from_env("playground")`           | 1,024 environments; Isaac Lab uses 4,096                                                       |
| `WarpSACConfig` | `WarpSACConfig.from_env("isaaclab")`             | GPU simulator settings plus Isaac Lab `n_step=3`                                               |
| `VSimbaConfig`  | `VSimbaConfig.from_env("dmc", "dog-run-visual")` | Defaults from the config class                                                                 |


For FastTD3 and FastSAC, set `num_train_env` in `from_env` so derived transition and buffer counts use
the chosen environment count. Explicit values take precedence: `WarpSACConfig.from_env("isaaclab", num_train_env=2048)`.

## Environments

`ApexRL.envs.create_envs(env_name, env_type, seed, ...)` returns `(train_envs, eval_envs, record_envs)` as
gymnasium `VectorEnv`s.


|                    | `env_type`                                                          |
| ------------------ | ------------------------------------------------------------------- |
| **CPU simulators** | `atari`, `mujoco`, `dmc`, `myosuite`, `humanoid_bench`, `metaworld` |
| **GPU simulators** | `playground`, `isaaclab`, `maniskill`, `mjlab`                      |


For single-task CPU environments in `mujoco`, `dmc`, `myosuite`, `humanoid_bench`, and `metaworld`, append
`-visual` to `env_name` to use pixels. The constructor strips the suffix, enables `rgb_array` rendering,
replaces state observations with RGB images, and stacks three frames as `[3, H, W, 3]`.


| Suite  | State observations | Visual observations     |
| ------ | ------------------ | ----------------------- |
| DMC    | `cheetah-run`      | `cheetah-run-visual`    |
| MuJoCo | `HalfCheetah-v5`   | `HalfCheetah-v5-visual` |


These DMC and MuJoCo examples produce observations shaped `[num_envs, 3, 84, 84, 3]` with dtype `uint8`.
The suffix is handled by `create_envs`, independently of `Config.from_env`. Atari and GPU simulators
use their own observation setup.

Atari training and evaluation run on [envpool](https://github.com/sail-sg/envpool) with the purejaxql settings:
life-loss episodes and reward clipping during training only, no sticky actions, up to 30 no-ops on reset, and
FIRE on reset. envpool cannot render, so video recording uses a single gymnasium ALE env configured the same way.

## Project layout

```
ApexRL/
├── agents/     # one folder per algorithm: config.py + torch/ + jax/
├── buffers/    # on-policy rollout buffers and off-policy replay buffers (torch & jax)
├── envs/       # create_envs() and per-simulator wrappers
├── model/      # shared layers, ensembles, policies, normalizers, backbones such as NatureCNN (torch & jax)
├── runners/    # OnPolicyRunner / OffPolicyRunner training loops
└── common/     # evaluation, logging, device helpers
```



## License

ApexRL is released under the [MIT License](LICENSE).

The implementations below draw on the listed projects. Licenses for adapted components are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).


| Component                | Source project                                                                                                           | License           |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------ | ----------------- |
| `ApexRL/agents/ppo/`     | [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl), [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) | MIT, BSD-3-Clause |
| `ApexRL/agents/pqn/`     | [mttga/purejaxql](https://github.com/mttga/purejaxql)                                                                    | Apache-2.0        |
| `ApexRL/agents/fasttd3/` | [younggyoseo/FastTD3](https://github.com/younggyoseo/FastTD3)                                                            | MIT               |
| `ApexRL/agents/fastsac/` | [amazon-far/holosoma](https://github.com/amazon-far/holosoma)                                                            | Apache-2.0        |
| `ApexRL/agents/warpsac/` | [wzhhasadream/warprl](https://github.com/wzhhasadream/warprl)                                                            | MIT               |
| `ApexRL/agents/v_simba/` | [DAVIAN-Robotics/V-Simba](https://github.com/DAVIAN-Robotics/V-Simba)                                                    | Apache-2.0        |




## Citation

If you use these algorithms, please cite the original papers.

**BibTeX**

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

@article{kim2026flashsac,
  title={FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control},
  author={Kim, Donghu and Lee, Youngdo and Park, Minho and Kim, Kinam and Nahendra, I Made Aswin and Seno, Takuma and Min, Sehee and Palenicek, Daniel and Vogt, Florian and Kragic, Danica and Peters, Jan and Choo, Jaegul and Lee, Hojoon},
  journal={arXiv preprint arXiv:2604.04539},
  year={2026}
}

@misc{wu2026warpsac,
  title         = {WarpSAC: Towards the Pinnacle of Scalable Off-policy RL by Rethinking Exploration and Exploitation},
  author        = {Wu, Zihao and Tang, Hongyao and Ma, Yi and Song, Huizhong and Li, Pengyi and Yuan, Yifu and Ni, Fei and Liu, Jinyi and Wei, Wei and Wang, Jianrong and Zheng, Yan and Hao, Jianye},
  year          = {2026},
  eprint        = {2608.24479},
  archivePrefix = {arXiv}
}

@article{kim2026vsimba,
  title   = {V-Simba: Unleashing the Architectural Potential of RL in Visual Continuous Control},
  author  = {Kim, Donghu and Lee, Youngdo and Lee, Hojoon and Obando-Ceron, Johan and Lee, Byungkun and Courville, Aaron and Castro, Pablo Samuel and Choo, Jaegul and Lyle, Clare},
  journal = {Reinforcement Learning Journal},
  year    = {2026}
}
```
