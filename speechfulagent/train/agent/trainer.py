from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import gymnasium as gym

from .replay_buffer import ReplayBuffer
from .wrappers import RewardWrapper
from speechfulagent.agent import Agent, Actor, Critic
from speechfulagent.dataclasses import *


class AgentTrainer:
    def __init__(
        self,
        env: gym.Env,
        objective: float,
        gamma: float,
        replay_buffer_size: int,
        replay_buffer_start_size: int,
        batch_size: int,
        learning_rate: float,
        sync_target_frames: int,
        ou_enable: bool,
        ou_mu: float,
        ou_theta: float,
        ou_sigma: float,
        ou_epsilon: float,
        logger = None
    ):
        self.env = RewardWrapper(env)

        self.objective = objective

        self.gamma = gamma

        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.replay_buffer_start_size = replay_buffer_start_size

        self.actor = Actor(env.observation_space.n, env.action_space.n)
        self.tgt_actor = Actor(env.observation_space.n, env.action_space.n)
        self.critic = Critic(env.observation_space.n, env.action_space.n)
        self.tgt_critic = Critic(env.observation_space.n, env.action_space.n)
        self.sync_target_frames = sync_target_frames
        self.batch_size = batch_size
        self.actor_optim = optim.Adam(params=self.actor.parameters(), lr=learning_rate)
        self.critic_optim = optim.Adam(params=self.critic.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

        self.ou_enabled = ou_enable
        self.ou_mu = ou_mu
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

        self.logger = logger

        self.agent = Agent()
        self.agent.actor = self.actor
        self.agent.critic = self.critic
    
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
        states_t = F.one_hot(torch.as_tensor(states), self.env.observation_space.n)
        actions_t = F.one_hot(torch.as_tensor(actions), self.env.action_space.n)
        rewards_t = torch.as_tensor(rewards)
        next_states_t = F.one_hot(torch.as_tensor(next_states), self.env.observation_space.n)
        dones_t = torch.as_tensor(dones)
        return states_t, actions_t, rewards_t, next_states_t, dones_t
    
    def train(self) -> Tuple[Agent, EnvInfo, AgentTrainInfo]:
        n_iter = 0
        total_rewards = []
        state, _ = self.env.reset()
        self.agent.reset()
        self.agent.reset_ou()
        self.agent.init_state(state)
        writer = SummaryWriter()
        while True:
            n_iter += 1
            exp = self.agent.step(self.env)
            self.replay_buffer.append(exp)
            if exp.done:
                reward = self.agent.total_reward
                total_rewards.append(reward)
                m_reward = np.mean(total_rewards[-100:])
                if self.logger:
                    self.logger.info(f"{n_iter}: done {len(total_rewards)} games, reward {m_reward:.3f}")
                writer.add_scalar("reward_100", m_reward, n_iter)
                writer.add_scalar("reward", reward, n_iter)

                if m_reward > self.objective:
                    if self.logger:
                        self.logger.info(f"Solved in {n_iter} iterations!")
                    break

                state, _ = self.env.reset()
                self.agent.reset()
                self.agent.init_state(state)
            
            if len(self.replay_buffer) < self.replay_buffer_start_size:
                continue
            
            batch = self.replay_buffer.sample(self.batch_size)
            states_t, actions_t, rewards_t, next_states_t, dones_t = self._batch_to_tensors(batch)

            # critic training
            self.critic_optim.zero_grad()
            q_v = self.critic(states_t, actions_t)
            next_actions_v = self.tgt_actor(next_states_t)
            next_q_v = self.tgt_critic(next_states_t, next_actions_v)
            next_q_v[dones_t] = 0.0
            q_ref_v = rewards_t.unsqueeze(dim=-1) + next_q_v * self.gamma
            critic_loss = F.mse_loss(q_v, q_ref_v.detach())
            critic_loss.backward()
            self.critic_optim.step()

            # actor training
            self.actor_optim.zero_grad()
            cur_actions_v = self.actor(states_t)
            actor_loss = -self.critic(states_t, cur_actions_v)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optim.step()

            if n_iter % self.sync_target_frames == 0:
                self.tgt_actor.load_state_dict(self.actor.state_dict())
                self.tgt_critic.load_state_dict(self.critic.state_dict())

        writer.close()

        env_info = EnvInfo(
            self.env.spec.id,
            int(self.env.observation_space.n),
            int(self.env.action_space.n)
        )
        train_info = AgentTrainInfo(
            n_iter,
            self.objective,
            self.gamma,
            len(self.replay_buffer),
            self.replay_buffer_start_size,
            self.batch_size,
            self.learning_rate,
            self.sync_target_frames,
            self.ou_enabled,
            self.ou_mu,
            self.ou_theta,
            self.ou_sigma,
            self.ou_epsilon
        )
        return self.agent, env_info, train_info