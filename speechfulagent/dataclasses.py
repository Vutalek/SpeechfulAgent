from dataclasses import dataclass, asdict
from typing import Tuple, Any

from numpy.typing import NDArray

from speechfulagent.types import *


@dataclass
class Experience:
    state: State
    action: Action | NDArray
    reward: float
    next_state: State
    done: bool

    dict = asdict

@dataclass
class EnvInfo:
    name: str
    n_observations: int | Tuple[Any]
    n_actions: int
    
    dict = asdict

@dataclass
class A3CTrainInfo:
    # iterations count
    n_iter: int
    # objective bound
    mean_objective: float
    # gamma
    gamma: float
    # optimization
    batch_size: int
    n_steps: int
    learning_rate: float
    # clipping
    clip_grad: float
    # asynchronous
    n_envs: int
    
    dict = asdict

@dataclass
class PPOTrainInfo:
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

    dict = asdict

@dataclass
class DDPGTrainInfo:
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
    # Ornshtein-Uhlenbeck process
    ou_enabled: bool
    ou_mu: float
    ou_theta: float
    ou_sigma: float
    ou_epsilon: float

    dict = asdict

@dataclass
class DQNTrainInfo:
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

    dict = asdict

BaseTrainInfo = DQNTrainInfo | A3CTrainInfo | DDPGTrainInfo | PPOTrainInfo

@dataclass
class ExplainerTrainInfo:
    #dataset
    pathfile: str
    max_length: int
    # iterations count
    n_iter: int
    # optimization
    batch_size: int
    learning_rate: float

    dict = asdict