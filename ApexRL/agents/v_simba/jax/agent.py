from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from gymnasium.vector import VectorEnv

from ....buffers.off_policy import Transition
from ....buffers.off_policy.numpy_lazy_frame_buffer import NumpyLazyFrameBuffer
from ....model.jax import Alpha, Network, RewardNormalizer
from ....model.jax import update_reward_normalizer
from ....model.jax.backbones import VSimbaVisionEncoder
from ...base_agent import OffPolicyAgent
from ..config import VSimbaConfig
from .get_action import get_eval_action, get_exploration_action
from .network import Actor, Critic
from .update import make_update


class VSimbaAgent(OffPolicyAgent):
    def __init__(self, envs: VectorEnv, cfg: VSimbaConfig):
        super().__init__(envs, cfg)
        if len(self.observation_shape) != 4 or self.action_is_discrete or self.num_envs != 1:
            raise ValueError("V-Simba requires one pixel environment with continuous actions")
        if not (np.allclose(self.action_space.low, -1) and np.allclose(self.action_space.high, 1)):
            raise ValueError("V-Simba requires actions rescaled to [-1, 1]")
        self.cfg.target_entropy = self.cfg.temp_target_entropy_coef * self.action_dim
        self._action_key, self._update_key = jax.random.split(jax.random.PRNGKey(self.cfg.seed))
        self._init_state()
        self._init_cached_fn()

    def _init_state(self):
        cfg = self.cfg
        rngs = nnx.Rngs(cfg.seed)
        compute_type = getattr(jnp, cfg.compute_type)
        self.replay_buffer = NumpyLazyFrameBuffer(
            self.observation_space, self.action_space, max_size=cfg.buffer_size,
            linear_decay_step=self.cfg.decay_step, max_n_step=cfg.n_step,
        )
        encoder = VSimbaVisionEncoder(
            self.observation_shape, rngs.fork(), num_channels=cfg.encoder_num_channels,
            num_blocks=cfg.encoder_num_blocks, conv_kernel_size=cfg.encoder_conv_kernel_size,
        )
        input_dim = encoder.get_output_dim()
        actor = Actor(input_dim, self.action_dim, rngs.fork(), cfg.actor_num_blocks, cfg.actor_hidden_dim, compute_type)
        critic = Critic(
            input_dim, self.action_dim, rngs.fork(), num_qs=2 if cfg.critic_use_cdq else 1,
            num_blocks=cfg.critic_num_blocks, hidden_dim=cfg.critic_hidden_dim, action_embed_dim=cfg.critic_action_embed_dim,
            c_shift=cfg.critic_c_shift, num_bins=cfg.critic_num_bins, min_v=cfg.critic_min_v, max_v=cfg.critic_max_v, compute_type=compute_type,
        )
        alpha = Alpha(cfg.temp_initial_value)
        tx = optax.adamw(cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.encoder = Network(encoder, nnx.Optimizer(encoder, tx, wrt=nnx.Param))
        self.actor = Network(actor, nnx.Optimizer(actor, tx, wrt=nnx.Param), forward_name="get_mean_action")
        self.critic = Network(critic, nnx.Optimizer(critic, tx, wrt=nnx.Param))
        self.alpha = Network(alpha, nnx.Optimizer(alpha, tx, wrt=nnx.Param))
        self.target_critic = Network(deepcopy(critic))
        self.reward_normalizer = None
        if cfg.normalize_rewards:
            normalizer = RewardNormalizer(self.num_envs, cfg.gamma, cfg.normalized_g_max)
            normalizer.g_rms.count.value = jnp.asarray(1e-4, dtype=jnp.float32)
            self.reward_normalizer = Network(normalizer)

    def _init_cached_fn(self):
        self._get_eval_fn = nnx.cached_partial(get_eval_action, self.encoder, self.actor)
        self._get_exploration_fn = nnx.cached_partial(get_exploration_action, self.encoder, self.actor)
        self._update_fn = nnx.cached_partial(make_update(self.cfg), self.encoder, self.actor, self.critic, self.target_critic, self.alpha, self.reward_normalizer)
        self._update_reward_normalizer = nnx.cached_partial(update_reward_normalizer, self.reward_normalizer)

    def get_action(self, observation: jax.Array | np.ndarray) -> np.ndarray:
        observation = jnp.asarray(observation).reshape((-1, *self.observation_shape))
        return np.asarray(self._get_eval_fn(observation))

    def get_exploration_action(self, observation: jax.Array | np.ndarray) -> np.ndarray:
        observation = jnp.asarray(observation).reshape((-1, *self.observation_shape))
        self._action_key, key = jax.random.split(self._action_key)
        return np.asarray(self._get_exploration_fn(observation, key))

    def process_transition(self, transition: Transition) -> None:
        if self.reward_normalizer is not None:
            self._update_reward_normalizer(transition.rewards, transition.terminations, transition.truncations)
        self.replay_buffer.add(transition)

    @property
    def can_update(self) -> bool:
        return self.replay_buffer.size >= self.cfg.learning_starts and self.replay_buffer.can_sample(self.cfg.n_step)

    def update(self) -> dict[str, float]:
        if not self.can_update:
            raise RuntimeError("Replay buffer is not ready for an update")
        sequence = jax.tree.map(jnp.asarray, self.replay_buffer.sample(self.cfg.batch_size, self.cfg.n_step))
        self._update_key, key = jax.random.split(self._update_key)
        info = self._update_fn(sequence, key)
        return {name: float(value) for name, value in info.items()}

    def save(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.encoder.save(checkpoint_dir / "encoder.ckpt")
        self.actor.save(checkpoint_dir / "actor.ckpt")
        self.critic.save(checkpoint_dir / "critic.ckpt")
        self.target_critic.save(checkpoint_dir / "target_critic.ckpt")
        self.alpha.save(checkpoint_dir / "alpha.ckpt")
        if self.reward_normalizer is not None:
            self.reward_normalizer.save(checkpoint_dir / "reward_normalizer.ckpt")

    def load(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.encoder.load(checkpoint_dir / "encoder.ckpt")
        self.actor.load(checkpoint_dir / "actor.ckpt")
        self.critic.load(checkpoint_dir / "critic.ckpt")
        self.target_critic.load(checkpoint_dir / "target_critic.ckpt")
        self.alpha.load(checkpoint_dir / "alpha.ckpt")
        if self.reward_normalizer is not None:
            self.reward_normalizer.load(checkpoint_dir / "reward_normalizer.ckpt")
        self._init_cached_fn()

    def save_encoder(self, checkpoint_dir: str | Path) -> None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.encoder.save(Path(checkpoint_dir) / "encoder.ckpt")

    def load_encoder(self, checkpoint_dir: str | Path) -> None:
        self.encoder.load(Path(checkpoint_dir) / "encoder.ckpt")
        self._init_cached_fn()

    def save_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.save(Path(checkpoint_dir) / "replay_buffer")

    def load_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.load(Path(checkpoint_dir) / "replay_buffer")

    def save_onnx(self, onnx_dir: str | Path) -> None:
        policy = nnx.Sequential(self.encoder.model, Network(self.actor.model, forward_name="get_mean_action"))
        Network(policy).save_onnx(Path(onnx_dir) / "policy.onnx", [(1, *self.observation_shape)])
