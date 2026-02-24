from typing import Dict, Any

import numpy as np
import torch
import gymnasium as gym

from speechfulagent.types import *
from speechfulagent.dataclasses import *
from speechfulagent.versioning import VersioningMixin
from .ddpgactor import Actor
from .ddpgcritic import Critic


class Agent(VersioningMixin):
    """DDPG agent. Only works in environments with both discrete action
    and observation spaces."""
    def __init__(
        self,
        ou_enable: bool=True,
        ou_mu: float=0.0,
        ou_theta: float=0.15,
        ou_sigma: float=0.2,
        ou_epsilon: float=1.0
    ):
        self.env_state = None
        self.total_reward = 0.0
        self.actor = None
        self.critic = None
        self.ou_enabled = ou_enable
        self.ou_mu = ou_mu
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon
        self.actions_state = None

    def reset(self):
        # resets the state and total reward of agent
        self.env_state = None
        self.total_reward = 0.0

    def reset_ou(self):
        self.actions_state = None

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
    
        # state_t = torch.as_tensor(self._ohe(self.env_state, env.observation_space.n))
        state_t = torch.as_tensor(self.env_state)
        state_t.unsqueeze_(0)
        mu_v = self.actor(state_t)
        actions = mu_v.squeeze(0)
        # Ornshtein-Uhlenbeck process
        if self.ou_enabled and self.ou_epsilon > 0.0:
            if self.actions_state is None:
                self.actions_state = np.zeros(actions.shape, dtype=np.float32)
            self.actions_state += self.ou_theta * (self.ou_mu - self.actions_state)
            self.actions_state += self.ou_sigma * np.random.normal(size=actions.shape)

            actions += self.ou_epsilon * self.actions_state

        # _, act_t = torch.max(actions, dim=0)
        # action = int(act_t.item())
        action = np.clip(actions, -1.0, 1.0)
        
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

        self.critic = Critic(data["environment"]["n_observations"], data["environment"]["n_actions"])
        self.critic.load_state_dict(state_dict["critic"])
