from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import gymnasium as gym

from .wrappers import RewardWrapper
from speechfulagent.agent import Agent, Actor, Critic
from speechfulagent.dataclasses import *


class AgentTrainer:
    def __init__(
        self,
        env: gym.Env,
        objective: float,
        gamma: float,
        gae_lambda: float,
        trajectory_size: int,
        epochs: int,
        eps: float,
        batch_size: int,
        learning_rate_actor: float,
        learning_rate_critic: float,
        logger = None
    ):
        self.env = RewardWrapper(env)

        self.objective = objective

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.trajectory = []
        self.trajectory_size = trajectory_size
        self.epochs = epochs
        self.eps = eps

        self.actor_net = Actor(env.observation_space.n, env.action_space.n)
        self.critic_net = Critic(env.observation_space.n)
        self.optim_actor = optim.Adam(params=self.actor_net.parameters(), lr=learning_rate_actor)
        self.optim_critic = optim.Adam(params=self.critic_net.parameters(), lr=learning_rate_critic)
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.batch_size = batch_size

        self.logger = logger

        self.agent = Agent()
        self.agent.actor = self.actor_net
        self.agent.critic = self.critic_net

    def _reference_advantage(
        self, 
        trajectory: List[Experience], 
        states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vals = self.critic_net(states)
        vals = vals.squeeze().data.numpy()
        # generalized advantage estimation
        last_gae = 0.0
        result_adv = []
        result_ref = []
        for val, next_val, exp in zip(
            reversed(vals[:-1]), reversed(vals[1:]), reversed(trajectory[:-1])
        ):
            if exp.done:
                delta = exp.reward - val
                last_gae = delta
            else:
                delta = exp.reward + self.gamma * next_val - val
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            result_adv.append(last_gae)
            result_ref.append(last_gae + val)
        advs = torch.FloatTensor(np.asarray(list(reversed(result_adv))))
        refs = torch.FloatTensor(np.asarray(list(reversed(result_ref))))
        return advs, refs
    
    def _batch_to_tensors(
        self, 
        batch: List[Experience]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        states, actions = [], []
        for e in batch:
            states.append(e.state)
            actions.append(e.action)
        states_t = F.one_hot(torch.as_tensor(states), self.env.observation_space.n)
        actions_t = torch.as_tensor(actions)
        return states_t, actions_t
    
    def train(self) -> Tuple[Agent, EnvInfo, AgentTrainInfo]:
        n_iter = 0
        total_rewards = []
        self.agent.reset()
        state, _ = self.env.reset()
        self.agent.init_state(state)
        writer = SummaryWriter()
        while True:
            n_iter += 1
            exp = self.agent.step(self.env)

            self.trajectory.append(exp)
            if exp.done:
                reward = self.agent.total_reward
                total_rewards.append(reward)
                m_reward = np.mean(total_rewards[-100:])
                if self.logger:
                    self.logger.info(
                        f"{n_iter}: done {len(total_rewards)} games, reward {m_reward:.3f}"
                    )
                writer.add_scalar("reward_100", m_reward, n_iter)
                writer.add_scalar("reward", reward, n_iter)

                if m_reward > self.objective:
                    if self.logger:
                        self.logger.info(f"Solved in {n_iter} iterations!")
                    break

                self.agent.reset()
                state, _ = self.env.reset()
                self.agent.init_state(state)
            
            if len(self.trajectory) < self.trajectory_size:
                continue
            
            states, actions = self._batch_to_tensors(self.trajectory)
            advantages, references = self._reference_advantage(self.trajectory, states)
            logits = self.actor_net(states)
            logprob_old = torch.gather(
                torch.log_softmax(logits, dim=-1),
                1,
                actions.unsqueeze(0).T
            ).squeeze()

            # standartization
            advantages = advantages - torch.mean(advantages)
            advantages /= torch.std(advantages)
            logprob_old = logprob_old[:-1].detach()

            for _ in range(self.epochs):
                for batch_offset in range(0, self.trajectory_size-1, self.batch_size):
                    batch_end = batch_offset + self.batch_size
                    batch_states = states[batch_offset:batch_end]
                    batch_actions = actions[batch_offset:batch_end]
                    batch_advs = advantages[batch_offset:batch_end].unsqueeze(-1)
                    batch_refs = references[batch_offset:batch_end]
                    batch_logprob_old = logprob_old[batch_offset:batch_end]

                    # critic
                    self.optim_critic.zero_grad()
                    value = self.critic_net(batch_states)
                    loss = F.mse_loss(value.squeeze(), batch_refs)
                    loss.backward()
                    self.optim_critic.step()

                    # actor
                    self.optim_actor.zero_grad()
                    batch_logits = self.actor_net(batch_states)
                    batch_logprob = torch.gather(
                        torch.log_softmax(batch_logits, dim=-1),
                        1,
                        batch_actions.unsqueeze(0).T
                    ).squeeze()
                    ratio = torch.exp(batch_logprob - batch_logprob_old)
                    surr_obj = batch_advs * ratio
                    clip_ratio = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps)
                    clip_surr_obj = batch_advs * clip_ratio
                    loss_policy = -torch.min(surr_obj, clip_surr_obj).mean()
                    loss_policy.backward()
                    self.optim_actor.step()
                    writer.add_scalar("actor loss", loss_policy, n_iter)
                    writer.add_scalar("critic loss", loss, n_iter)
            self.trajectory.clear()
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
            self.gae_lambda,
            self.trajectory_size,
            self.epochs,
            self.eps,
            self.batch_size,
            self.learning_rate_actor,
            self.learning_rate_critic
        )
        return self.agent, env_info, train_info
