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
    n_observatons: int
    n_actions: int
    
    dict = asdict

@dataclass
class AgentTrainInfo:
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
    #early stopping
    early_stopping_steps: int

    dict = asdict

@dataclass
class ExplainerTrainInfo:
    # iterations count
    n_iter: int
    # optimization
    batch_size: int
    learning_rate: float

    dict = asdict