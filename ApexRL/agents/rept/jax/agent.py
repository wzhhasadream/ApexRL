from copy import deepcopy
from pathlib import Path
import gymnasium as gym
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
from ....model.jax.backbones import VSimbaVisionEncoder, DRQVisionEncoder, DRQV2VisionEncoder
from ....common.jax import default_device
from ...base_agent import OffPolicyAgent
from ..config import RePTConfig
from .get_action import (
    get_eval_action,
    get_exploration_action,
    update_reward_normalizer,
)
from .network import Actor, Critic
from .update import make_update 

class RePTAgent(OffPolicyAgent):
    """JAX off-policy RePrent agent with reward/value heads and NCE critic."""

    def __init__(self, envs: VectorEnv, cfg: RePTConfig) -> None:
        if len(envs.single_observation_space.shape) not in (1, 4):
            raise ValueError("RePT requires flat states or [F, H, W, C] pixel observations.")
        if not isinstance(envs.action_space, gym.spaces.Box):
            raise ValueError("RePT only support continuous actions")
        super().__init__(envs, cfg)
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
        num_updates = max(
            1,
            int(
                self.cfg.total_timesteps
                / self.num_envs
                * self.cfg.grad_step_per_interaction_step
            ),
        )
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
        self.vision_encoder = None
        if self.image_obs:
            self.replay_buffer = NumpyLazyFrameBuffer(
                self.observation_space, self.action_space, max_size=self.cfg.buffer_size,
                linear_decay_step=self.cfg.decay_step, max_n_step=self.cfg.n_step,
                num_envs=self.num_envs,
            )
            if self.cfg.vision_encoder == "drq_dot":
                encoder_model = DRQVisionEncoder(self.observation_shape, rngs.fork())
            elif self.cfg.vision_encoder == "drqv2":
                encoder_model = DRQV2VisionEncoder(self.observation_shape, rngs.fork())
            elif self.cfg.vision_encoder == "vsimba":
                encoder_model = VSimbaVisionEncoder(self.observation_shape, rngs.fork())
            else:
                raise ValueError(f"Unknown vision_encoder: {self.cfg.vision_encoder}")
            self.vision_encoder = Network(
                encoder_model,
                nnx.Optimizer(encoder_model, optax.adam(critic_schedule), wrt=nnx.Param),
            )
            critic_obs_dim = encoder_model.get_output_dim()
            actor_obs_dim = self.actor_obs_shape[0] if self.asymmetric_obs else critic_obs_dim
        else:
            self.replay_buffer = JaxBuffer.create(
                observation_space=self.observation_space,
                action_space=self.action_space,
                max_size=self.cfg.buffer_size,
                linear_decay_step=self.cfg.decay_step,
                num_envs=self.num_envs,
                use_approximate_sampling=self.cfg.buffer_device == "cpu",
                device=self.cfg.buffer_device,
            )
            actor_obs_dim, critic_obs_dim = self.actor_obs_shape[0], self.critic_obs_shape[0]

        actor_model = Actor(
            actor_obs_dim,
            self.action_dim,
            rngs.fork(),
            hidden_dim=self.cfg.actor_hidden_dim,
            num_mlp_blocks=self.cfg.actor_num_blocks,
            action_is_discrete=self.action_is_discrete,
            action_low=-1.0,
            action_high=1.0,
            use_bias=self.cfg.use_bias,
            compute_type=compute_type,
        )
        critic_model = Critic(
            critic_obs_dim,
            self.action_dim,
            hidden_dim=self.cfg.critic_hidden_dim,
            rngs=rngs.fork(split=self.cfg.num_q),
            num_mlp_blocks=self.cfg.num_encoder_blocks,
            num_trunk_blocks=self.cfg.critic_num_trunk_blocks,
            use_bias=self.cfg.use_bias,
            compute_type=compute_type,
            num_head=self.cfg.num_head,
        )



        self.actor = Network(
            actor_model,
            nnx.Optimizer(
                actor_model,
                optax.adam(policy_schedule),
                wrt=nnx.Param,
            ),
            forward_name="get_deterministic_action",
        )
        self.critic = Network(
            critic_model,
            nnx.Optimizer(
                critic_model,
                optax.adam(critic_schedule),
                wrt=nnx.Param,
            ),
        )

        alpha_model = Alpha()
        self.alpha = Network(alpha_model, nnx.Optimizer(alpha_model, optax.adam(policy_schedule), wrt=nnx.Param))
        self.target_critic = Network(
            deepcopy(self.critic.model),
            source_model=self.critic.model,
            tau=self.cfg.tau,
        )
        self.reward_normalizer = (
            Network(RewardNormalizer(self.num_envs, self.cfg.gamma))
            if self.cfg.normalize_rewards
            else None
        )

        if self.cfg.actor_normalize_parameters:
            self.actor.project_param()
        if self.cfg.critic_normalize_parameters:
            self.critic.project_param()
            self.target_critic.project_param()

        self._learner = default_device()

    def _init_cached_fn(self) -> None:

        self._get_eval_fn = nnx.cached_partial(get_eval_action, self.actor, self.vision_encoder, self.asymmetric_obs)
        self._get_exploration_fn = nnx.cached_partial(get_exploration_action, self.actor, self.asymmetric_obs, self.vision_encoder)
        self._update_fn = nnx.cached_partial(
            make_update(self.cfg), self.critic, self.target_critic, self.vision_encoder,
            self.actor, self.alpha, self.reward_normalizer,
        )
        if self.reward_normalizer is not None:
            self._update_reward_normalizer = nnx.cached_partial(
                update_reward_normalizer,
                self.reward_normalizer,
            )
        else:
            self._update_reward_normalizer = None

    def get_action(self, observation: jax.Array | np.ndarray) -> np.ndarray:
        observation = jnp.asarray(observation).reshape(
            (-1, *self.observation_shape)
        )
        actions = self._get_eval_fn(observation)
        return np.asarray(actions)

    def get_exploration_action(
        self,
        observation: jax.Array | np.ndarray,
    ) -> np.ndarray:
        observation = jnp.asarray(observation).reshape(
            (-1, *self.observation_shape)
        )
        self._action_key, action_key = jax.random.split(self._action_key)
        (
            self.cached_key,
            actions,
            self.repeat_n,
            self.repeat_count,
        ) = self._get_exploration_fn(
            observation,
            self.repeat_n,
            self.repeat_count,
            self.cached_key,
            action_key,
        )
        return np.asarray(actions)

    def process_transition(self, transition: Transition) -> None:
        if self._update_reward_normalizer is not None:
            self._update_reward_normalizer(
                transition.rewards,
                transition.terminations,
                transition.truncations,
            )
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
        sequence = jax.tree.map(lambda value: jax.device_put(value, self._learner), sequence)

        self._update_key, update_key = jax.random.split(self._update_key)
        do_policy = self._critic_updates % self.cfg.policy_frequency == 0
        do_target = (
            (self._critic_updates + 1) % self.cfg.target_frequency == 0
        )
        info = self._update_fn(sequence, update_key, do_policy, do_target)
        self._critic_updates += 1
        return {name: float(value) for name, value in info.items()}

    def save(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.actor.save(checkpoint_dir / "actor.ckpt")
        self.critic.save(checkpoint_dir / "critic.ckpt")
        self.target_critic.save(checkpoint_dir / "target_critic.ckpt")
        self.alpha.save(checkpoint_dir / "alpha.ckpt")
        if self.vision_encoder is not None:
            self.vision_encoder.save(checkpoint_dir / "vision_encoder.ckpt")
        if self.reward_normalizer is not None:
            self.reward_normalizer.save(checkpoint_dir / "reward_normalizer.ckpt")

    def load(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.actor.load(checkpoint_dir / "actor.ckpt")
        self.critic.load(checkpoint_dir / "critic.ckpt")
        self.target_critic.load(checkpoint_dir / "target_critic.ckpt")
        self.alpha.load(checkpoint_dir / "alpha.ckpt")
        if self.vision_encoder is not None:
            self.vision_encoder.load(checkpoint_dir / "vision_encoder.ckpt")
        normalizer_checkpoint = checkpoint_dir / "reward_normalizer.ckpt"
        if self.reward_normalizer is not None and normalizer_checkpoint.is_dir():
            self.reward_normalizer.load(normalizer_checkpoint)
        self._init_cached_fn()

    def save_encoder(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        encoder = self.critic.model.encoder
        if self.vision_encoder is not None:
            encoder = nnx.Dict(vision=self.vision_encoder.model, phi=encoder)
        Network(encoder).save(checkpoint_dir / "encoder.ckpt")

    def load_encoder(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        encoder = self.critic.model.encoder
        if self.vision_encoder is not None:
            encoder = nnx.Dict(vision=self.vision_encoder.model, phi=encoder)
        Network(encoder).load(checkpoint_dir / "encoder.ckpt")
        self._init_cached_fn()

    def save_buffer(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir) / "replay_buffer"
        self.replay_buffer.save(checkpoint_dir)

    def load_buffer(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir) / "replay_buffer"
        if self.image_obs:
            self.replay_buffer.load(checkpoint_dir)
        else:
            self.replay_buffer = self.replay_buffer.load(checkpoint_dir)

    def save_onnx(self, onnx_dir: str | Path) -> None:
        onnx_dir = Path(onnx_dir)
        policy = self.actor
        if self.vision_encoder is not None:
            # Compose only for export; training keeps the encoder and actor separate.
            policy = Network(nnx.Sequential(self.vision_encoder.model, self.actor))
        policy.save_onnx(onnx_dir / "policy.onnx", [(1, *self.actor_obs_shape)])
