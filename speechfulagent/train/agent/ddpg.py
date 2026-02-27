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
from speechfulagent.agent import DDPGAgent
from speechfulagent.agent.net import DDPGActor, DDPGCritic
from .replay_buffer import ReplayBuffer


class DDPGTrainer(BaseTrainer):
    def __init__(
        self,
        env: gym.Env,
        objective: float,
        actor: Optional[torch.nn.Module]=None,
        critic: Optional[torch.nn.Module]=None,
        gamma: float=0.99,
        replay_buffer_size: int=100000,
        replay_buffer_start_size: int=10000,
        batch_size: int=64,
        learning_rate: float=1e-4,
        alpha_sync: float=0.999,
        ou_enable: bool=True,
        ou_mu: float=0.0,
        ou_theta: float=0.15,
        ou_sigma: float=0.2,
        ou_epsilon: float=1.0,
        writer: Optional[SummaryWriter]=None,
        logger=None,
        seed: int=70
    ):
        super().__init__(seed)
        self.env = env
        self.agent = DDPGAgent(self.env, ou_enable, ou_mu, ou_theta, ou_sigma, ou_epsilon, self.seed)

        self.objective = objective

        self.gamma = gamma

        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.replay_buffer_start_size = replay_buffer_start_size

        if self.agent.is_obs_cont:
            if self.agent.obs_shape is not None:
                obs = self.agent.obs_shape[0]
            else:
                obs = 0
        else:
            obs = self.agent.obs_n

        if self.agent.act_shape is not None:
            act = self.agent.act_shape[0]
        else:
            act = 0
            
        if actor is not None:
            self.actor = actor
        else:
            self.actor = DDPGActor(obs, act)

        if critic is not None:
            self.critic = critic
        else:
            self.critic = DDPGCritic(obs, act)

        self.tgt_actor = deepcopy(self.actor)
        self.tgt_critic = deepcopy(self.critic)
        self.alpha = alpha_sync
        self.batch_size = batch_size
        self.actor_optim = optim.Adam(params=self.actor.parameters(), lr=learning_rate)
        self.critic_optim = optim.Adam(params=self.critic.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

        self.ou_enabled = ou_enable
        self.ou_mu = ou_mu
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

        self.writer = writer
        self.logger = logger

        self.agent.set_actor(self.actor)
        self.agent.set_critic(self.critic)
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
        actions_t = torch.as_tensor(np.array(actions))
        rewards_t = torch.as_tensor(rewards)
        if self.agent.is_obs_cont:
            next_states_t = torch.as_tensor(np.array(next_states))
        else:
            next_states_t = F.one_hot(torch.as_tensor(next_states), self.agent.obs_n)
        dones_t = torch.as_tensor(dones)
        return states_t, actions_t, rewards_t, next_states_t, dones_t
    
    def _alphasync(self, train, target):
        state = train.state_dict()
        tgt_state = target.state_dict()
        for k, v in state.items():
            tgt_state[k] = self.alpha * tgt_state[k] + (1 - self.alpha) * v
        target.load_state_dict(tgt_state)
    
    def train(self) -> Tuple[DDPGAgent, EnvInfo, DDPGTrainInfo]:
        n_iter = 0
        total_rewards = []
        self.agent.reset()
        self.agent.reset_ou()
        while True:
            n_iter += 1
            exp = self.agent.step()
            # self.agent.ou_epsilon *= 0.999
            self.replay_buffer.append(exp)
            if exp.done:
                reward = self.agent.total_reward
                total_rewards.append(reward)
                m_reward = np.mean(total_rewards[-100:])
                if self.logger:
                    self.logger.info(f"{n_iter}: done {len(total_rewards)} games, reward {m_reward:.3f}")
                if self.writer:
                    self.writer.add_scalar("reward_100", m_reward, n_iter)
                    self.writer.add_scalar("reward", reward, n_iter)

                if m_reward > self.objective:
                    if self.logger:
                        self.logger.info(f"Solved in {n_iter} iterations!")
                    break

                self.agent.reset()
                self.agent.reset_ou()
            
            if len(self.replay_buffer) < self.replay_buffer_start_size:
                continue
            
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = self._batch_to_tensors(batch)

            # critic training
            self.critic_optim.zero_grad()
            q_v = self.critic(states, actions)
            next_actions_v = self.tgt_actor(next_states)
            next_q_v = self.tgt_critic(next_states, next_actions_v)
            mask = (~dones).float().unsqueeze(-1)
            q_ref_v = rewards.unsqueeze(-1) + mask * next_q_v * self.gamma
            critic_loss = F.mse_loss(q_v, q_ref_v.detach())
            if self.writer:
                self.writer.add_scalar("q values", q_v.mean().item(), n_iter)
                self.writer.add_scalar("critic_loss", critic_loss.item(), n_iter)
            critic_loss.backward()
            self.critic_optim.step()

            # actor training
            self.actor_optim.zero_grad()
            cur_actions_v = self.actor(states)
            actor_loss = -self.critic(states, cur_actions_v)
            actor_loss = actor_loss.mean()
            if self.writer:
                self.writer.add_scalar("actor_loss", actor_loss.item(), n_iter)
            actor_loss.backward()
            self.actor_optim.step()

            # alphasync
            self._alphasync(self.actor, self.tgt_actor)
            self._alphasync(self.critic, self.tgt_critic)

        env_info = EnvInfo(
            self.env.spec.id,
            self.agent.obs_shape if self.agent.is_obs_cont else self.agent.obs_n,
            self.agent.act_shape
        )
        train_info = DDPGTrainInfo(
            n_iter,
            self.objective,
            self.gamma,
            len(self.replay_buffer),
            self.replay_buffer_start_size,
            self.batch_size,
            self.learning_rate,
            self.alpha,
            self.ou_enabled,
            self.ou_mu,
            self.ou_theta,
            self.ou_sigma,
            self.ou_epsilon,
            self.seed
        )
        self.agent.eval()
        return self.agent, env_info, train_info
