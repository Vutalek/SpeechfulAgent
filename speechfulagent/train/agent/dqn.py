from typing import List, Tuple, Optional
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import gymnasium as gym

from .base_trainer import BaseTrainer
from speechfulagent.dataclasses import *
from speechfulagent.agent import DQNAgent
from speechfulagent.agent.net import MLPDQN
from .replay_buffer import ReplayBuffer


class DQNTrainer(BaseTrainer):
    def __init__(
        self,
        env: gym.Env,
        objective: float,
        net: Optional[torch.nn.Module]=None,
        gamma: float=0.99,
        replay_buffer_size: int=50000,
        replay_buffer_start_size: int=1000,
        batch_size: int=64,
        learning_rate: float=1e-4,
        sync_target_frames: int=100,
        epsilon_decay_last_frame: int=60000,
        epsilon_decay_start: float=1.0,
        epsilon_decay_final: float=0.01,
        writer: Optional[SummaryWriter]=None,
        logger=None
    ):
        super().__init__()
        self.env = env
        self.agent = DQNAgent(self.env)

        self.objective = objective

        self.gamma = gamma

        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.replay_buffer_start_size = replay_buffer_start_size

        if net is not None:
            self.train_net = net
        else:
            if self.agent.is_obs_cont:
                if self.agent.obs_shape is not None:
                    shape = self.agent.obs_shape[0]
                else:
                    shape = 0
                self.train_net = MLPDQN(shape, self.agent.act_n)
            else:
                self.train_net = MLPDQN(self.agent.obs_n, self.agent.act_n)
        self.target_net = deepcopy(self.train_net)

        self.sync_target_frames = sync_target_frames
        self.batch_size = batch_size
        self.optim = optim.Adam(params=self.train_net.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate
        
        self.epsilon_decay_last_frame = epsilon_decay_last_frame
        self.epsilon_decay_start = epsilon_decay_start
        self.epsilon_decay_final = epsilon_decay_final

        self.writer = writer
        self.logger = logger

        self.agent.set_model(self.train_net)
        self.agent.train()
    
    def _batch_to_tensors(
        self, 
        batch: List[Experience]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for e in batch:
            states.append(e.state)
            actions.append(e.action)
            rewards.append(e.reward)
            next_states.append(e.next_state)
            dones.append(e.done)
        if self.agent.is_obs_cont:
            states_t = torch.as_tensor(np.array(states))
        else:
            states_t = F.one_hot(torch.as_tensor(states), self.agent.obs_n)
        actions_t = torch.as_tensor(actions)
        rewards_t = torch.as_tensor(rewards)
        if self.agent.is_obs_cont:
            next_states_t = torch.as_tensor(np.array(next_states))
        else:
            next_states_t = F.one_hot(torch.as_tensor(next_states), self.agent.obs_n)
        dones_t = torch.as_tensor(dones)
        return states_t, actions_t, rewards_t, next_states_t, dones_t
    
    def _loss(self, batch: List[Experience]) -> torch.Tensor:
        states_t, actions_t, rewards_t, next_states_t, dones_t = self._batch_to_tensors(batch)
        q_values = self.train_net(states_t).gather(
            1, actions_t.unsqueeze(-1)
        ).squeeze(-1)
        with torch.no_grad():
            next_state_values = self.target_net(next_states_t).max(1)[0]
            next_state_values[dones_t] = 0.0
            next_state_values = next_state_values.detach()

        expected_q_values = next_state_values * self.gamma + rewards_t
        return F.mse_loss(q_values, expected_q_values)
    
    def train(self) -> Tuple[DQNAgent, EnvInfo, DQNTrainInfo]:
        n_iter = 0
        total_rewards = []
        epsilon = self.epsilon_decay_start
        self.agent.reset()
        while True:
            n_iter += 1
            epsilon = max(
                self.epsilon_decay_final,
                self.epsilon_decay_start - n_iter / self.epsilon_decay_last_frame
            )

            self.agent.epsilon = epsilon
            exp = self.agent.step()
            self.replay_buffer.append(exp)
            if exp.done:
                reward = self.agent.total_reward
                total_rewards.append(reward)
                m_reward = np.mean(total_rewards[-100:])
                if self.logger:
                    self.logger.info(
                        f"{n_iter}: done {len(total_rewards)} games, reward {m_reward:.3f}, " + \
                        f"epsilon {epsilon:.2f}"
                    )
                if self.writer:
                    self.writer.add_scalar("epsilon", epsilon, n_iter)
                    self.writer.add_scalar("reward_100", m_reward, n_iter)
                    self.writer.add_scalar("reward", reward, n_iter)

                if m_reward > self.objective:
                    if self.logger:
                        self.logger.info(f"Solved in {n_iter} iterations!")
                    break

                self.agent.reset()
            
            if len(self.replay_buffer) < self.replay_buffer_start_size:
                continue
            if n_iter % self.sync_target_frames == 0:
                self.target_net.load_state_dict(self.train_net.state_dict())
            
            self.optim.zero_grad()
            batch = self.replay_buffer.sample(self.batch_size)
            loss = self._loss(batch)
            loss.backward()
            self.optim.step()

        env_info = EnvInfo(
            self.env.spec.id,
            self.agent.obs_shape if self.agent.is_obs_cont else self.agent.obs_n,
            self.agent.act_n
        )
        train_info = DQNTrainInfo(
            n_iter,
            self.objective,
            self.gamma,
            len(self.replay_buffer),
            self.replay_buffer_start_size,
            self.batch_size,
            self.learning_rate,
            self.sync_target_frames,
            self.epsilon_decay_last_frame,
            self.epsilon_decay_start,
            self.epsilon_decay_final
        )
        return self.agent, env_info, train_info