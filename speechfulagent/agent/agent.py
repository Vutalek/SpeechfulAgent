import os

import numpy as np
import torch
import gymnasium as gym
import yaml
from yamlmaker import generate

from speechfulagent.types import *
from speechfulagent.dataclasses import *
from speechfulagent.versioning import VersioningMixin
from .net import DQN


class Agent(VersioningMixin):
    def __init__(self):
        self.env_state = None
        self.total_reward = 0.0
        self.net = None
        self.version = None

    def reset(self):
        self.env_state = None
        self.total_reward = 0.0

    def init_state(self, state: State):
        self.env_state = state

    def _ohe(self, x, size):
        enc = np.zeros(size)
        enc[x] = 1
        return enc
    
    @torch.no_grad()
    def step(self, env: gym.Env, epsilon: float = 0.0) -> Experience:
        self.get_version()
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_t = torch.as_tensor(self._ohe(self.env_state, env.observation_space.n))
            state_t.unsqueeze_(0)
            q_values = self.net(state_t)
            _, act_t = torch.max(q_values, dim=1)
            action = int(act_t.item())
        
        next_state, reward, is_done, is_trunc, _ = env.step(action)
        self.total_reward += reward

        old_state = self.env_state
        self.env_state = next_state

        done = is_done or is_trunc
        exp = Experience(
            old_state,
            action,
            reward,
            next_state,
            done
        )
        return exp
    
    def get_version(self) -> str | None:
        """Returns current version of agent's model"""
        if self.version is not None:
            return self.version
        else:
            raise RuntimeError("Model not loaded!")
    
    def save_model(self, dir: str, env: EnvInfo, train: TrainInfo):
        version = self.get_next_version(dir)
        path = dir + '/' + version
        os.mkdir(path)
        torch.save(self.net.state_dict(), path + '/weights.dat')
        info = {
            "version": version,
            "n_params": sum(p.numel() for p in self.net.parameters()),
            "environment": env.dict(),
            "training": train.dict()
        }
        generate(info, path + "/info")

    def load_model(self, dir: str, version = "latest"):
        known = self.known_versions(dir)
        if version == "latest":
            version = self.get_latest(dir)
        if version not in known:
            raise RuntimeError("Unknown version!")
        self.version = version

        with open(dir + '/' + version + '/' + "info.yml", "rt") as f:
            data = yaml.safe_load(f)

        state_dict = torch.load(
            dir + '/' + version + '/' + "weights.dat",
            map_location=lambda stg, _: stg,
            weights_only=True
        )
        self.net = DQN(data["environment"]["input_shape"], data["environment"]["n_actions"])
        self.net.load_state_dict(state_dict)