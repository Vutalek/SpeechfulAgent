from dataclasses import dataclass, asdict

from speechfulagent.types import *


@dataclass
class Experience:
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool

    dict = asdict

@dataclass
class EnvInfo:
    name: str
    input_shape: int
    n_actions: int
    
    dict = asdict

@dataclass
class TrainInfo:
    # iterations count
    n_iter: int
    # objective bound
    mean_default_bound: float
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
    # epsilon decayt
    epsilon_decay_last_frame: int
    epsilon_decay_start: float
    epsilon_decay_final: float

    dict = asdict