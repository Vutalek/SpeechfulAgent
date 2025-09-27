import os
import re
import typing as tt
from collections import defaultdict
import pickle as pkl

import gymnasium as gym
from yamlmaker import generate

from speechfulagent.types import *
from speechfulagent.versioning import VersioningMixin


class Agent(VersioningMixin):
    def __init__(
        self,
        env: gym.Env,
        gamma: float,
        alpha: float
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.env = env
        self.state, _ = self.env.reset()
        self.values: tt.Dict[ValuesKey, float] = defaultdict(float)
        self.version = None

    def sample_env(self) -> tt.Tuple[State, Action, float, State]:
        """Gets sars' tuple from random action in environment"""
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, is_trunc, _ = self.env.step(action)
        if is_done or is_trunc:
            self.state, _ = self.env.reset()
        else:
            self.state = new_state
        return old_state, action, float(reward), new_state
    
    def get_best_action(self, state: State) -> tt.Tuple[float, Action]:
        """Return largest q-value for state and corresponding action"""
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action
    
    def value_update(self, state: State, action: Action, reward: float, next_state: State):
        """Updating q-value for state, action"""
        # value of next state
        best_val, _ = self.get_best_action(next_state)
        # value of state
        new_val = reward + self.gamma * best_val
        old_val = self.values[(state, action)]
        self.values[(state, action)] = old_val * (1 - self.alpha) + new_val * self.alpha

    def play_episode(self, env: gym.Env) -> float:
        total_reward = 0.0
        state, _ = env.reset()
        while True:
            _, action = self.get_best_action(state)
            new_state, reward, is_done, is_trunc, _ = env.step(action)
            total_reward += float(reward)
            if is_done or is_trunc:
                break
            state = new_state
        return total_reward
    
    def get_version(self) -> str | None:
        """Returns current version of agent's model"""
        if self.version is not None:
            return self.version
        else:
            raise RuntimeError("Model not loaded!")
    
    def save_model(self, dir: str):
        version = self.get_next_version(dir)
        path = dir + '/' + version
        os.mkdir(path)
        with open(path + "/q_table.pkl", 'wb') as f:
            pkl.dump(self.values, f)
        info = {
            "version": version,
            "environment": self.env.spec.id,
            "gamma": self.gamma,
            "alpha": self.alpha
        }
        generate(info, path + "/info")

    def load_model(self, dir: str, version = "latest"):
        known = self.known_versions(dir)
        if version == "latest":
            version = self.get_latest(dir)
        if version not in known:
            raise RuntimeError("Unknown version!")
        self.version = version
        with open(dir + '/' + version + '/' + "q_table.pkl", 'rb') as f:
            self.values = pkl.load(f)