"""Dataclasses for project."""

from dataclasses import dataclass, asdict
from typing import Tuple, Any, List

from speechfulagent.types import State, Action


@dataclass
class Experience:
    """Experience from environment."""
    state: State
    action: Action
    reward: float
    next_state: State | None
    done: bool

    dict = asdict

@dataclass
class EnvInfo:
    """Basic information about environment."""
    name: str
    n_observations: int | Tuple[Any, ...]
    n_actions: int | Tuple[Any, ...]

    dict = asdict

@dataclass
class DQNTrainInfo:
    """Hyperparameters and some results from DQN training."""
    # iterations count
    n_iter: int
    # objective bound
    mean_objective: float
    # gamma
    gamma: float
    # replay buffer
    replay_buffer_size: int
    replay_buffer_start_size: int
    # optimization
    batch_size: int
    learning_rate: float
    # target network
    sync_target_frames: int
    # epsilon decay
    epsilon_decay_last_frame: int
    epsilon_decay_start: float
    epsilon_decay_final: float

    seed: int

    dict = asdict

@dataclass
class A3CTrainInfo:
    """Hyperparameters and some results from A3C training."""
    # iterations count
    n_iter: int
    # objective bound
    mean_objective: float
    # gamma
    gamma: float
    # optimization
    worker_batch_size: int
    train_batch_size: int
    n_steps: int
    learning_rate: float
    # clipping
    clip_grad: float
    # asynchronous
    n_envs: int

    seed: int

    dict = asdict

@dataclass
class DDPGTrainInfo:
    """Hyperparameters and some results from DDPG training."""
    # iterations count
    n_iter: int
    # objective bound
    mean_objective: float
    # gamma
    gamma: float
    # replay buffer
    replay_buffer_size: int
    replay_buffer_start_size: int
    # optimization
    batch_size: int
    learning_rate: float
    # target network
    alpha_sync: float
    # Ornshtein-Uhlenbeck process
    ou_enabled: bool
    ou_mu: float
    ou_theta: float
    ou_sigma: float
    ou_epsilon: float

    seed: int

    dict = asdict

@dataclass
class PPOTrainInfo:
    """Hyperparameters and some results from PPO training."""
    # iterations count
    n_iter: int
    # objective bound
    mean_objective: float
    # gamma
    gamma: float
    # ppo + gae
    gae_lambda: float
    trajectory_size: int
    epochs: int
    eps: float
    # optimization
    batch_size: int
    learning_rate_actor: float
    learning_rate_critic: float

    seed: int

    dict = asdict

@dataclass
class ExplainerTrainInfo:
    """Hyperparameters and some results from explainer training."""
    #dataset
    pathfile: str
    max_length: int
    # iterations count
    n_iter: int
    # optimization
    batch_size: int
    learning_rate: float

    dict = asdict

@dataclass
class R3LTrainInfo:
    # dataset
    pathfile: str
    validation_fraction: float
    # llm
    llm_model_name: str
    llm_tokenizer_name: str
    llm_think_start: List[int]
    llm_think_end: List[int]
    llm_text_start: List[int]
    # state_encoder
    modules: List[int]
    module_size: int
    hidden_size: int
    projected_size: int
    # devices
    encoder_device: str
    llm_device: str
    # iterations count
    n_iter: int
    # optimization
    learning_rate: float
    weight_decay: float
    early_stopping: int

    seed: int

    dics = asdict

BaseTrainInfo = DQNTrainInfo | A3CTrainInfo | DDPGTrainInfo | PPOTrainInfo | R3LTrainInfo