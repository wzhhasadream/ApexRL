from copy import deepcopy
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from gymnasium.vector import VectorEnv

from ....buffers.off_policy import Transition
from ....buffers.off_policy.jax_buffer import JaxBuffer
from ....buffers.off_policy.numpy_lazy_frame_buffer import NumpyLazyFrameBuffer
from ....model.jax import Alpha, Network, RewardNormalizer
from ....common.jax import default_device
from ...base_agent import OffPolicyAgent
from ..config import RePTConfig
from .get_action import (
    get_eval_action,
    get_exploration_action,
    update_reward_normalizer,
)
from .network import Actor, Critic, Encoder
from .update import make_update

class RePTAgent(OffPolicyAgent):
    """JAX off-policy RePrent agent with reward/value heads and NCE critic."""

    def __init__(self, envs: VectorEnv, cfg: RePTConfig) -> None:
        if len(envs.single_observation_space.shape) not in (1, 4):
            raise ValueError("RePT requires flat states or [F, H, W, C] pixel observations.")
        super().__init__(envs, cfg)
        if self.action_is_discrete or self.asymmetric_obs:
            raise ValueError("RePT requires continuous actions and shared actor/critic observations")
        self._init_state()
        self._init_cached_fn()

        self._action_key, self._sample_key, self._update_key = jax.random.split(
            jax.random.PRNGKey(cfg.seed), 3
        )
        self.cached_key = jax.random.PRNGKey(0)
        self.repeat_count = jnp.array(0, dtype=jnp.int32)
        self.repeat_n = jnp.array(1, dtype=jnp.int32)
        self._critic_updates = 0

    def _init_state(self) -> None:
        compute_type = getattr(jnp, self.cfg.compute_type)
        num_updates = max(1, int(self.cfg.total_timesteps / self.num_train_env * self.cfg.grad_step_per_interaction_step))
        policy_schedule = optax.cosine_decay_schedule(
            init_value=self.cfg.policy_lr,
            decay_steps=num_updates,
            alpha=self.cfg.end_lr / self.cfg.policy_lr,
        )
        critic_schedule = optax.cosine_decay_schedule(
            init_value=self.cfg.q_lr,
            decay_steps=num_updates,
            alpha=self.cfg.end_lr / self.cfg.q_lr,
        )

        rngs = nnx.Rngs(self.cfg.seed)
        if self.image_obs:
            self.replay_buffer = NumpyLazyFrameBuffer(
                self.observation_space, self.action_space, max_size=self.cfg.buffer_size,
                linear_decay_step=self.cfg.decay_step, max_n_step=self.cfg.n_step,
                num_envs=self.num_train_env,
            )
        else:
            self.replay_buffer = JaxBuffer.create(
                observation_space=self.observation_space,
                action_space=self.action_space,
                max_size=self.cfg.buffer_size,
                linear_decay_step=self.cfg.decay_step,
                num_envs=self.num_train_env,
                use_approximate_sampling=self.cfg.buffer_device == "cpu",
                device=self.cfg.buffer_device,
            )

        encoder_model = Encoder(
            self.observation_shape, self.action_dim, rngs.fork(),
            num_blocks=self.cfg.num_encoder_blocks, z_s_dim=self.cfg.z_s_dim, z_sa_dim=self.cfg.z_sa_dim,
            use_bias=self.cfg.use_bias, compute_type=compute_type, vision_encoder=self.cfg.vision_encoder,
        )

        actor_model = Actor(
            self.cfg.z_s_dim,
            self.action_dim,
            rngs.fork(),
            hidden_dim=self.cfg.actor_hidden_dim,
            num_blocks=self.cfg.num_actor_blocks,
            action_low=jnp.asarray(self.action_low),
            action_high=jnp.asarray(self.action_high),
            use_bias=self.cfg.use_bias,
            compute_type=compute_type,
        )
        critic_model = Critic(
            z_sa_dim=self.cfg.z_sa_dim,
            hidden_dim=self.cfg.critic_hidden_dim,
            rngs=rngs.fork(split=self.cfg.num_q),
            num_blocks=self.cfg.num_critic_blocks,
            use_bias=self.cfg.use_bias,
            compute_type=compute_type,
            num_head=self.cfg.num_head,
        )
        self.encoder = Network(encoder_model, nnx.Optimizer(encoder_model, optax.adam(critic_schedule), wrt=nnx.Param))
        self.actor = Network(
            actor_model, nnx.Optimizer(actor_model, optax.adam(policy_schedule), wrt=nnx.Param),
            forward_name="get_deterministic_action",
        )
        self.critic = Network(critic_model, nnx.Optimizer(critic_model, optax.adam(critic_schedule), wrt=nnx.Param))
        alpha_model = Alpha()
        self.alpha = Network(alpha_model, nnx.Optimizer(alpha_model, optax.adam(policy_schedule), wrt=nnx.Param))
        self.reward_normalizer = (
            Network(RewardNormalizer(self.num_train_env, self.cfg.gamma))
            if self.cfg.normalize_rewards
            else None
        )

        if self.cfg.actor_normalize_parameters:
            self.actor.project_param()
        if self.cfg.critic_normalize_parameters:
            self.encoder.project_param()
            self.critic.project_param()
        self.target_encoder = Network(deepcopy(encoder_model), source_model=encoder_model, tau=self.cfg.tau)
        self.target_critic = Network(deepcopy(critic_model), source_model=critic_model, tau=self.cfg.tau)

        self._learner = default_device()

    def _init_cached_fn(self) -> None:
        self._get_eval_fn = nnx.cached_partial(get_eval_action, self.encoder, self.actor)
        self._get_exploration_fn = nnx.cached_partial(get_exploration_action, self.encoder, self.actor)
        self._update_fn = nnx.cached_partial(
            make_update(self.cfg), self.encoder, self.critic, self.target_encoder, self.target_critic,
            self.actor, self.alpha, self.reward_normalizer,
        )
        self._update_reward_normalizer = nnx.cached_partial(update_reward_normalizer, self.reward_normalizer)

    def get_action(self, observation: jax.Array | np.ndarray) -> np.ndarray:
        observation = jnp.asarray(observation).reshape((-1, *self.observation_shape))
        return np.asarray(self._get_eval_fn(observation))

    def get_exploration_action(self, observation: jax.Array | np.ndarray) -> np.ndarray:
        observation = jnp.asarray(observation).reshape((-1, *self.observation_shape))
        self._action_key, action_key = jax.random.split(self._action_key)
        self.cached_key, actions, self.repeat_n, self.repeat_count = self._get_exploration_fn(
            observation, self.repeat_n, self.repeat_count, self.cached_key, action_key,
        )
        return np.asarray(actions)

    def process_transition(self, transition: Transition) -> None:
        self._update_reward_normalizer(transition.rewards, transition.terminations, transition.truncations)
        if self.image_obs:
            self.replay_buffer.add(transition)
        else:
            self.replay_buffer = self.replay_buffer.add(transition)

    @property
    def can_update(self) -> bool:
        return bool(
            self.replay_buffer.size >= self.cfg.learning_starts
            and self.replay_buffer.can_sample(self.cfg.n_step)
        )

    def update(self) -> dict[str, float]:
        if not self.can_update:
            raise RuntimeError("Replay buffer is not ready for an update.")

        if self.image_obs:
            sequence = self.replay_buffer.sample(self.cfg.batch_size, self.cfg.n_step)
        else:
            self._sample_key, sample_key = jax.random.split(self._sample_key)
            sequence = self.replay_buffer.sample(sample_key, self.cfg.batch_size, self.cfg.n_step)
        sequence = jax.device_put(sequence, self._learner)

        self._update_key, update_key = jax.random.split(self._update_key)
        do_policy = self._critic_updates % self.cfg.policy_frequency == 0
        do_target = (self._critic_updates + 1) % self.cfg.target_frequency == 0
        info = self._update_fn(sequence, update_key, do_policy, do_target)
        self._critic_updates += 1
        return {name: float(value) for name, value in info.items()}

    def save(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.encoder.save(checkpoint_dir / "encoder.ckpt")
        self.target_encoder.save(checkpoint_dir / "target_encoder.ckpt")
        self.actor.save(checkpoint_dir / "actor.ckpt")
        self.critic.save(checkpoint_dir / "critic.ckpt")
        self.target_critic.save(checkpoint_dir / "target_critic.ckpt")
        self.alpha.save(checkpoint_dir / "alpha.ckpt")
        if self.reward_normalizer is not None:
            self.reward_normalizer.save(checkpoint_dir / "reward_normalizer.ckpt")

    def load(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.encoder.load(checkpoint_dir / "encoder.ckpt")
        self.target_encoder.load(checkpoint_dir / "target_encoder.ckpt")
        self.actor.load(checkpoint_dir / "actor.ckpt")
        self.critic.load(checkpoint_dir / "critic.ckpt")
        self.target_critic.load(checkpoint_dir / "target_critic.ckpt")
        self.alpha.load(checkpoint_dir / "alpha.ckpt")
        normalizer_checkpoint = checkpoint_dir / "reward_normalizer.ckpt"
        if self.reward_normalizer is not None and normalizer_checkpoint.is_dir():
            self.reward_normalizer.load(normalizer_checkpoint)
        self._init_cached_fn()

    def save_encoder(self, checkpoint_dir: str | Path) -> None:
        self.encoder.save(Path(checkpoint_dir) / "encoder.ckpt")

    def load_encoder(self, checkpoint_dir: str | Path) -> None:
        self.encoder.load(Path(checkpoint_dir) / "encoder.ckpt")
        nnx.update(self.target_encoder.model, nnx.state(self.encoder.model))
        self._init_cached_fn()

    def save_buffer(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir).absolute() / "replay_buffer"
        self.replay_buffer.save(checkpoint_dir)

    def load_buffer(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir).absolute() / "replay_buffer"
        if self.image_obs:
            self.replay_buffer.load(checkpoint_dir)
        else:
            self.replay_buffer = self.replay_buffer.load(checkpoint_dir)

    def save_onnx(self, onnx_dir: str | Path) -> None:
        onnx_dir = Path(onnx_dir)
        state_encoder = Network(self.encoder.model, forward_name="embed_s")
        policy = Network(nnx.Sequential(state_encoder, self.actor))
        policy.save_onnx(onnx_dir / "policy.onnx", [(1, *self.actor_obs_shape)])
