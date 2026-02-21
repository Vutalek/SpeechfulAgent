from typing import Dict, Any

import numpy as np
import torch
import gymnasium as gym

from speechfulagent.types import *
from speechfulagent.dataclasses import *
from speechfulagent.versioning import VersioningMixin
from .actor import Actor
from .critic import Critic


class Agent(VersioningMixin):
    """PPO agent. Only works in environments with both discrete action
    and observation spaces."""
    def __init__(self):
        self.env_state = None
        self.total_reward = 0.0
        self.actor = None
        self.critic = None

    def reset(self):
        # resets the state and total reward of agent
        self.env_state = None
        self.total_reward = 0.0

    def init_state(self, state: State):
        # initializes first state
        self.env_state = state

    def _ohe(self, x, size):
        enc = np.zeros(size)
        enc[x] = 1
        return enc
    
    @torch.no_grad()
    def step(self, env: gym.Env) -> Experience:
        """Agent's step in environment.

        Firstly, agent must be initialized with init_state() method.
        """
        # checking if model is loaded
        if self.actor is None:
            raise RuntimeError("Model not loaded!")
        # checking if state is not None
        if self.env_state is None:
            raise RuntimeError("Uninitialized environment!")

        state_t = torch.as_tensor(self._ohe(self.env_state, env.observation_space.n))
        state_t.unsqueeze_(0)
        policy = self.actor(state_t)
        probs = torch.softmax(policy, dim=-1)
        action = int(torch.multinomial(probs, 1).item())
        
        next_state, reward, is_done, is_trunc, _ = env.step(action)
        self.total_reward += float(reward)

        old_state = self.env_state
        self.env_state = next_state

        done = is_done or is_trunc
        exp = Experience(
            old_state,
            action,
            float(reward),
            next_state,
            done
        )
        return exp
    
    def _save_model(self, path: str, version: str, *args, **kwargs) -> Dict[str, Any]:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict()
            },
            path + '/' + 'weights.dat'
        )
        env: EnvInfo = kwargs.get("env")
        train: AgentTrainInfo = kwargs.get("train")
        info = {
            "version": version,
            "n_params_actor": sum(p.numel() for p in self.actor.parameters()),
            "n_params_critic": sum(p.numel() for p in self.critic.parameters()),
            "environment": env.dict(),
            "training": train.dict()
        }
        return info

    def _load_model(self, path: str, data: Dict[str, Any], *args, **kwargs):
        state_dict = torch.load(path + '/' + "weights.dat")
        self.actor = Actor(data["environment"]["n_observations"], data["environment"]["n_actions"])
        self.actor.load_state_dict(state_dict["actor"])

        self.critic = Critic(data["environment"]["n_observations"])
        self.critic.load_state_dict(state_dict["critic"])