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
    n_observations: int
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
    # optimization
    batch_size: int
    learning_rate_actor: float
    learning_rate_critic: float

    dict = asdict

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