from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import gymnasium as gym

from .base_trainer import BaseTrainer
from speechfulagent.dataclasses import *
from speechfulagent.agent import PPOAgent
from speechfulagent.agent.net import DiscretePPOActor, ContinuousPPOActor, PPOCritic


class PPOTrainer(BaseTrainer):
    def __init__(
        self,
        env: gym.Env,
        objective: float,
        actor: Optional[torch.nn.Module]=None,
        critic: Optional[torch.nn.Module]=None,
        gamma: float=0.95,
        gae_lambda: float=0.95,
        trajectory_size: int=2049,
        epochs: int=10,
        eps: float=0.2,
        batch_size: int=64,
        learning_rate_actor: float=0.0003,
        learning_rate_critic: float=0.0003,
        writer: Optional[SummaryWriter]=None,
        logger=None,
        seed: int=70
    ):
        super().__init__(seed)
        self.env = env
        self.agent = PPOAgent(self.env, self.seed)

        self.objective = objective

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.trajectory = []
        self.trajectory_size = trajectory_size
        self.epochs = epochs
        self.eps = eps

        if self.agent.is_obs_cont:
            if self.agent.obs_shape is not None:
                obs = self.agent.obs_shape[0]
            else:
                obs = 0
        else:
            obs = self.agent.obs_n
        if self.agent.is_act_cont:
            if self.agent.act_shape is not None:
                act = self.agent.act_shape[0]
            else:
                act = 0
        else:
            act = self.agent.act_n

        if actor is not None:
            self.actor = actor
        else:
            if self.agent.is_act_cont:
                self.actor = ContinuousPPOActor(obs, act)
            else:
                self.actor = DiscretePPOActor(obs, act)
        
        if critic is not None:
            self.critic = critic
        else:
            self.critic = PPOCritic(obs)

        self.actor_optim = optim.Adam(params=self.actor.parameters(), lr=learning_rate_actor)
        self.critic_optim = optim.Adam(params=self.critic.parameters(), lr=learning_rate_critic)
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.batch_size = batch_size

        self.writer = writer
        self.logger = logger

        self.agent.set_actor(self.actor)
        self.agent.set_critic(self.critic)
        self.agent.train()

    def _reference_advantage(
        self, 
        trajectory: List[Experience], 
        states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vals = self.critic(states)
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
        if self.agent.is_obs_cont:
            states_t = torch.as_tensor(np.array(states))
        else:
            states_t = F.one_hot(torch.as_tensor(states), self.agent.obs_n)

        if self.agent.is_act_cont:
            actions_t = torch.as_tensor(np.array(actions))
        else:
            actions_t = [int(e) for e in actions]
            actions_t = torch.LongTensor(actions)

        return states_t, actions_t
    
    def train(self) -> Tuple[PPOAgent, EnvInfo, PPOTrainInfo]:
        n_iter = 0
        total_rewards = []
        self.agent.reset()
        while True:
            n_iter += 1
            exp = self.agent.step()

            self.trajectory.append(exp)
            if exp.done:
                reward = self.agent.total_reward
                total_rewards.append(reward)
                m_reward = np.mean(total_rewards[-100:])
                if self.logger:
                    self.logger.info(
                        f"{n_iter}: done {len(total_rewards)} games, reward {m_reward:.3f}"
                    )
                if self.writer:
                    self.writer.add_scalar("reward_100", m_reward, n_iter)
                    self.writer.add_scalar("reward", reward, n_iter)

                if m_reward > self.objective:
                    if self.logger:
                        self.logger.info(f"Solved in {n_iter} iterations!")
                    break

                self.agent.reset()
            
            if len(self.trajectory) < self.trajectory_size:
                continue
            
            states, actions = self._batch_to_tensors(self.trajectory)
            advantages, references = self._reference_advantage(self.trajectory, states)
            if self.agent.is_act_cont:
                mu = self.actor(states)
                p1 = - ((mu - actions) ** 2) / (2*torch.exp(self.actor.logstd).clamp(min=1e-3))
                p2 = - torch.log(torch.sqrt(2 * torch.pi * torch.exp(self.actor.logstd)))
                logprob_old = p1 + p2
            else:
                logits = self.actor(states)
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
                    self.critic_optim.zero_grad()
                    value = self.critic(batch_states)
                    loss = F.mse_loss(value.squeeze(), batch_refs)
                    loss.backward()
                    self.critic_optim.step()

                    # actor
                    self.actor_optim.zero_grad()
                    if self.agent.is_act_cont:
                        batch_mu = self.actor(batch_states)
                        p1 = - ((batch_mu - batch_actions) ** 2) / (2*torch.exp(self.actor.logstd).clamp(min=1e-3))
                        p2 = - torch.log(torch.sqrt(2 * torch.pi * torch.exp(self.actor.logstd)))
                        batch_logprob = p1 + p2
                    else:
                        batch_logits = self.actor(batch_states)
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
                    self.actor_optim.step()
                    if self.writer:
                        self.writer.add_scalar("actor loss", loss_policy, n_iter)
                        self.writer.add_scalar("critic loss", loss, n_iter)
            self.trajectory.clear()

        env_info = EnvInfo(
            self.env.spec.id,
            self.agent.obs_shape if self.agent.is_obs_cont else self.agent.obs_n,
            self.agent.act_shape if self.agent.is_act_cont else self.agent.act_n
        )
        train_info = PPOTrainInfo(
            n_iter,
            self.objective,
            self.gamma,
            self.gae_lambda,
            self.trajectory_size,
            self.epochs,
            self.eps,
            self.batch_size,
            self.learning_rate_actor,
            self.learning_rate_critic,
            self.seed
        )
        self.agent.eval()
        return self.agent, env_info, train_info
