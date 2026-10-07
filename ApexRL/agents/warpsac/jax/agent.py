from copy import deepcopy
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from optax import adam, cosine_decay_schedule
from ....buffers.off_policy import Transition
from ....buffers.off_policy.jax_buffer import JaxBuffer
from ..config import WarpSACConfig
from ....model.jax import Alpha, Network, RewardNormalizer
from ....common.jax import default_device
from ...base_agent import OffPolicyAgent
from .network import FlashSACActor, FlashSACDoubleCritic
from gymnasium.vector import VectorEnv
from .get_action import (
    get_eval_action,
    get_exploration_action,
    update_reward_normalizer,
)
from .update import make_update_warpsac
from pathlib import Path


class WarpSACAgent(OffPolicyAgent):
    def __init__(
        self,
        envs: VectorEnv,
        cfg: WarpSACConfig,
    ) -> None:
        super().__init__(envs, cfg)
        if self.action_is_discrete or len(self.observation_shape) != 1 or len(self.actor_obs_shape) != 1:
            raise ValueError("WarpSAC requires continuous actions and flat observations")
        self.learner_device = default_device()
        self._num_steps = 0

        self._init_train_state()
        self._init_cached_fn()

        self._action_key, self._update_key, self._sample_key = jax.random.split(
            jax.random.PRNGKey(cfg.seed), 3
        )
        self._obs_cache = (None, None)

    def _init_cached_fn(self):
        update_fn = make_update_warpsac(self.cfg)
        self._update_fn = nnx.cached_partial(
            update_fn,
            self.critic,
            self.actor,
            self.alpha,
            self.target_critic,
            self.reward_normalizer,
        )
        self._get_action_fn = nnx.cached_partial(
            get_eval_action,
            self.actor,
            self.asymmetric_obs,
        )
        self._get_exploration_action_fn = nnx.cached_partial(
            get_exploration_action,
            self.actor,
            self.asymmetric_obs,
        )
        self._update_reward_normalizer = nnx.cached_partial(
            update_reward_normalizer,
            self.reward_normalizer,
        )

    def _init_train_state(
        self,
    ) :
        compute_type = getattr(jnp, self.cfg.compute_type)
        num_critic_updates = max(1, int(self.cfg.total_timesteps / self.num_envs * self.cfg.grad_step_per_interaction_step))
        end_lr = self.cfg.end_lr
        policy_lr = self.cfg.policy_lr
        q_lr = self.cfg.q_lr

        rngs = nnx.Rngs(self.cfg.seed)
        self.replay_buffer = JaxBuffer.create(
            action_space=self.action_space,
            observation_space=self.observation_space,
            num_envs=self.num_envs,
            max_size=self.cfg.buffer_size,
            linear_decay_step=self.cfg.decay_step,
            use_approximate_sampling=self.cfg.buffer_device == "cpu",
            device=self.cfg.buffer_device,
        )

        actor_model = FlashSACActor(
            self.actor_obs_shape,
            self.action_dim,
            rngs.fork(),
            hidden_dim=self.cfg.actor_hidden_dim,
            num_blocks=self.cfg.actor_num_blocks,
            action_high=jnp.asarray(self.action_high),
            action_low=jnp.asarray(self.action_low),
            use_bias=self.cfg.use_bias,
            compute_type=compute_type,
        )
        critic_model = FlashSACDoubleCritic(
            self.critic_obs_shape,
            self.action_dim,
            rngs.fork(split=self.cfg.num_q),
            hidden_dim=self.cfg.critic_hidden_dim,
            num_blocks=self.cfg.critic_num_blocks,
            num_head=self.cfg.num_head,
            use_bias=self.cfg.use_bias,
            compute_type=compute_type,
            dist_type=(
                "scalar"
                if self.cfg.num_head == 1
                else self.cfg.dist_type
            ),
        )
        self.actor = Network(
            actor_model,
            nnx.Optimizer(
                actor_model,
                adam(cosine_decay_schedule(policy_lr, num_critic_updates, end_lr / policy_lr)),
                wrt=nnx.Param,
            ),
            forward_name="get_mean_action",
        )
        self.critic = Network(
            critic_model,
            nnx.Optimizer(
                critic_model,
                adam(cosine_decay_schedule(q_lr, num_critic_updates, end_lr / q_lr)),
                wrt=nnx.Param,
            ),
        )
        alpha_model = Alpha()
        self.alpha = Network(
            alpha_model,
            nnx.Optimizer(
                alpha_model,
                adam(cosine_decay_schedule(policy_lr, num_critic_updates, end_lr / policy_lr)),
                wrt=nnx.Param,
            ),
        )
        if self.cfg.actor_normalize_parameters:
            self.actor.project_param()
        if self.cfg.critic_normalize_parameters:    
            self.critic.project_param()
        self.target_critic = Network(
            deepcopy(self.critic.model),
            None,
            source_model=self.critic.model,
            tau=self.cfg.tau,
        )
        self.reward_normalizer = (
            Network(RewardNormalizer(self.num_envs, self.cfg.gamma))
            if self.cfg.normalize_rewards
            else None
        )

        self.cached_key = jax.random.PRNGKey(0)
        self.repeat_count = jnp.array(0, dtype=jnp.int32)
        self.repeat_n = jnp.array(1, dtype=jnp.int32)
        self.critic_grad_updates = 0


    def get_action(self, obs: jax.Array | np.ndarray) -> np.ndarray:
        obs = jnp.asarray(obs, jnp.float32).reshape((-1,) + self.observation_shape)
        actions = self._get_action_fn(obs)
        return np.asarray(actions)

    def get_exploration_action(self, obs: jax.Array | np.ndarray) -> np.ndarray:
        obs_device = jnp.asarray(obs, jnp.float32, device=self.learner_device).reshape((-1,) + self.observation_shape)
        # Reuse this device copy when the runner stores the same observation.
        self._obs_cache = (obs, obs_device)
        self._action_key, action_key = jax.random.split(self._action_key, 2)
        self.cached_key, actions, self.repeat_n, self.repeat_count = self._get_exploration_action_fn(
            obs_device,
            self.repeat_n,
            self.repeat_count,
            self.cached_key,
            action_key,
        )
        return np.asarray(actions)

    def process_transition(self, transition: Transition) -> None:
        if self._obs_cache[0] is transition.observations and self._obs_cache[1].device == self.replay_buffer.observations.device:
            transition = transition._replace(observations=self._obs_cache[1])
        self._update_reward_normalizer(
            transition.rewards,
            transition.terminations,
            transition.truncations,
        )
        self.replay_buffer = self.replay_buffer.add(transition)
        self._obs_cache = (None, None)
        self._num_steps += 1

    @property
    def can_update(self) -> bool:
        return min(self._num_steps, self.replay_buffer.capacity) * self.num_envs >= self.cfg.learning_starts and self._num_steps >= self.cfg.n_step

    def update(self) -> dict[str, float]:
        if not self.can_update:
            raise RuntimeError("Replay buffer is not ready for an update")
        self._sample_key, sample_key = jax.random.split(self._sample_key, 2)
        sequence = self.replay_buffer.sample(sample_key, self.cfg.batch_size, self.cfg.n_step)
        sequence = jax.device_put(sequence, self.learner_device)
        self._update_key, update_key = jax.random.split(self._update_key)
        update_actor = self.critic_grad_updates % self.cfg.policy_frequency == 0
        self.critic_grad_updates += 1
        update_target = self.critic_grad_updates % self.cfg.target_frequency == 0
        info = self._update_fn(update_key, sequence, update_actor=update_actor, update_target=update_target)

        return {name: float(value) for name, value in jax.device_get(info).items()}


    def save(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.actor.save(checkpoint_dir / "actor.ckpt")
        self.critic.save(checkpoint_dir / "critic.ckpt")
        self.target_critic.save(checkpoint_dir / "target_critic.ckpt")
        self.alpha.save(checkpoint_dir / "alpha.ckpt")
        if self.reward_normalizer is not None:
            self.reward_normalizer.save(checkpoint_dir / "reward_normalizer.ckpt")

    def save_onnx(self, path: str | Path) -> None:
        Network(nnx.Sequential(self.actor)).save_onnx(path, [(1, *self.actor_obs_shape)], output_names=("actions",))


    def load(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.actor.load(checkpoint_dir / "actor.ckpt")
        self.critic.load(checkpoint_dir / "critic.ckpt")
        self.target_critic.load(checkpoint_dir / "target_critic.ckpt")
        self.alpha.load(checkpoint_dir / "alpha.ckpt")
        normalizer_checkpoint_dir = checkpoint_dir / "reward_normalizer.ckpt"
        if self.reward_normalizer is not None and normalizer_checkpoint_dir.is_dir():
            self.reward_normalizer.load(normalizer_checkpoint_dir)
        self._init_cached_fn()

    def save_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer.save(Path(checkpoint_dir).absolute() / "replay_buffer")

    def load_buffer(self, checkpoint_dir: str | Path) -> None:
        self.replay_buffer = self.replay_buffer.load(Path(checkpoint_dir).absolute() / "replay_buffer")
        self._num_steps = int(self.replay_buffer.steps)
