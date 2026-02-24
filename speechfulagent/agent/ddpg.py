from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from speechfulagent.types import *
from speechfulagent.dataclasses import *
from .base_agent import BaseAgent


class DDPGAgent(BaseAgent):
    """DDPG agent"""
    def __init__(
        self, 
        env: gym.Env,
        ou_enable: bool=True,
        ou_mu: float=0.0,
        ou_theta: float=0.15,
        ou_sigma: float=0.2,
        ou_epsilon: float=1.0,
        seed: int=70
    ):
        super().__init__(env, seed)
        self.actor: Optional[torch.nn.Module] = None
        self.critic: Optional[torch.nn.Module] = None

        self.ou_enabled = ou_enable
        self.ou_mu = ou_mu
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon
        self.action_state = None

    def reset_ou(self):
        self.action_state = None
    
    def _step(self) -> Experience:
        if self.actor is None:
            raise RuntimeError("Model not loaded!")
        # checking if state is not None
        if self.env_state is None:
            raise RuntimeError("Uninitialized environment!")
    
        if self.is_obs_cont:
            state = torch.as_tensor(self.env_state)
        else:
            state = F.one_hot(torch.as_tensor(self.env_state), self.obs_n)
        state.unsqueeze_(0)

        mu = self.actor(state)
        action = mu.data.numpy()
        if self.training and self.ou_enabled and self.ou_epsilon > 0.0:
            # Ornshtein-Uhlenbeck process
            if self.action_state is None:
                self.action_state = np.zeros(action.shape, dtype=np.float32)
            self.action_state += self.ou_theta * (self.ou_mu - self.action_state)
            self.action_state += self.ou_sigma * np.random.normal(size=action.shape)

            action += self.ou_epsilon * self.action_state
        if self.is_act_cont:
            action = np.clip(action, -1.0, 1.0)
        else:
            _, act_idx = torch.max(action, dim=1)
            action = int(act_idx.item())
        
        next_state, reward, is_done, is_trunc, _ = self.env.step(action)
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
        # checking if model is not None
        if self.actor is None or self.critic is None:
            raise RuntimeError("Nothing to save!")
        torch.save(
            {
                "actor": self.actor,
                "critic": self.critic
            },
            path + '/' + 'model.pth'
        )
        env: EnvInfo = kwargs.get("env", EnvInfo)
        train: DDPGTrainInfo = kwargs.get("train", DDPGTrainInfo)
        info = {
            "version": version,
            "n_params_actor": sum(p.numel() for p in self.actor.parameters()),
            "n_params_critic": sum(p.numel() for p in self.critic.parameters()),
            "environment": env.dict(),
            "training": train.dict()
        }
        return info

    def _load_model(self, path: str, data: Dict[str, Any], *args, **kwargs):
        models = torch.load(path + '/' + "model.pth")
        self.actor = models["actor"]
        self.critic = models["critic"]

    def set_actor(self, actor: torch.nn.Module):
        """Sets local actor for agent"""
        self.actor = actor

    def set_critic(self, critic: torch.nn.Module):
        """Sets local critic for agent"""
        self.actor = critic
