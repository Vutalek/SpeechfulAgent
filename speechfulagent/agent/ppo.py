from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from speechfulagent.types import *
from speechfulagent.dataclasses import *
from .base_agent import BaseAgent


class PPOAgent(BaseAgent):
    """PPO agent"""
    def __init__(self, env: gym.Env, seed: int=70):
        super().__init__(env, seed)
        self.actor: Optional[torch.nn.Module] = None
        self.critic: Optional[torch.nn.Module] = None
    
    def _step(self) -> Experience:
        # checking if model is loaded
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

        if self.is_act_cont:
            mu = self.actor(state)
            mu = mu.data.numpy()
            logstd = self.actor.logstd.data.numpy()
            noise = np.random.normal(size=logstd.shape)
            action = mu + np.exp(logstd) * noise
            action = np.clip(action, -1.0, 1.0)
        else:
            policy = self.actor(state)
            probs = torch.softmax(policy, dim=-1)
            action = int(torch.multinomial(probs, 1).item())
        
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
            path + '/' + 'weights.dat'
        )
        env: EnvInfo = kwargs.get("env", EnvInfo)
        train: PPOTrainInfo = kwargs.get("train", PPOTrainInfo)
        info = {
            "version": version,
            "n_params_actor": sum(p.numel() for p in self.actor.parameters()),
            "n_params_critic": sum(p.numel() for p in self.critic.parameters()),
            "environment": env.dict(),
            "training": train.dict()
        }
        return info

    def _load_model(self, path: str, data: Dict[str, Any], *args, **kwargs):
        models = torch.load(path + '/' + "weights.dat")
        self.actor = models["actor"]

        self.critic = models["critic"]

    def set_actor(self, actor: torch.nn.Module):
        """Sets local actor for agent"""
        self.actor = actor

    def set_critic(self, critic: torch.nn.Module):
        """Sets local critic for agent"""
        self.actor = critic
