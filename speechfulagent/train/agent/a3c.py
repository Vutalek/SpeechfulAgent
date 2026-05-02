"""Module with A3C trainer."""

from collections import deque
from typing import Tuple, Optional
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard.writer import SummaryWriter
import gymnasium as gym

from .base_trainer import BaseTrainer
from speechfulagent.dataclasses import Experience, EnvInfo, A3CTrainInfo
from speechfulagent.agent import A2CAgent
from speechfulagent.agent.net import DiscreteA2C, ContinuousA2C


class A3CTrainer(BaseTrainer):
    """A3C trainer for trainging A2C."""
    def __init__(
        self,
        env: gym.Env,
        objective: float,
        net: Optional[torch.nn.Module]=None,
        gamma: float=0.99,
        entropy_beta: float=0.01,
        clip_grad: float=0.1,
        n_envs: int=4,
        n_steps: int=4,
        worker_batch_size: int=64,
        train_batch_size: int=2,
        learning_rate: float=1e-3,
        writer: Optional[SummaryWriter]=None,
        logger=None,
        seed: int=70
    ):
        super().__init__(seed)
        self.env = env
        self.agent = A2CAgent(env, self.seed)

        self.objective = objective

        self.gamma = gamma

        self.entropy_beta = entropy_beta
        self.clip_grad = clip_grad

        self.n_envs = n_envs
        self.pool = []

        self.n_steps = n_steps

        if net is not None:
            self.train_net = net
        else:
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
                self.train_net = ContinuousA2C(obs, act)
            else:
                act = self.agent.act_n
                self.train_net = DiscreteA2C(obs, act)

        self.optim = optim.Adam(params=self.train_net.parameters(), lr=learning_rate)
        self.worker_batch_size = worker_batch_size
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate

        self.writer = writer
        self.logger = logger

        self.agent.set_model(self.train_net)
        self.agent.train()

    def train(self) -> Tuple[A2CAgent, EnvInfo, A3CTrainInfo]:
        mp.freeze_support()
        mp.set_start_method("spawn")
        self.train_net.share_memory()
        q = mp.Queue(maxsize=self.n_envs)
        self.pool = []
        for pid in range(self.n_envs):
            process = mp.Process(
                target=worker_function,
                args=(
                    pid,
                    self.env,
                    self.objective,
                    self.train_net,
                    self.gamma,
                    self.entropy_beta,
                    self.clip_grad,
                    self.n_steps,
                    self.worker_batch_size,
                    q,
                    self.writer is not None,
                    self.logger
                )
            )
            process.start()
            self.pool.append(process)

        n_iter = 0
        grads_accum = None
        try:
            while True:
                grads = q.get()
                if grads is None:
                    break
                n_iter += 1

                if grads_accum is None:
                    grads_accum = grads
                else:
                    for tgt_grad, grad in zip(grads_accum, grads):
                        tgt_grad += grad

                if n_iter % self.train_batch_size == 0:
                    for param, grad in zip(self.train_net.parameters(), grads_accum):
                        param.grad = torch.FloatTensor(grad)

                    torch.nn.utils.clip_grad_norm_(self.train_net.parameters(), self.clip_grad)
                    self.optim.step()
                    grads_accum = None
        finally:
            for proc in self.pool:
                proc.terminate()
                proc.join()

        env_info = EnvInfo(
            self.env.spec.id,
            self.agent.obs_shape if self.agent.is_obs_cont else self.agent.obs_n,
            self.agent.act_shape if self.agent.is_act_cont else self.agent.act_n
        )
        train_info = A3CTrainInfo(
            n_iter,
            self.objective,
            self.gamma,
            self.worker_batch_size,
            self.train_batch_size,
            self.n_steps,
            self.learning_rate,
            self.clip_grad,
            self.n_envs,
            self.seed
        )
        self.agent.eval()
        return self.agent, env_info, train_info

def worker_function(
    pid: int,
    env: gym.Env,
    objective: float,
    net: torch.nn.Module,
    gamma: float,
    entropy_beta: float,
    clip_grad: float,
    n_steps: int,
    batch_size: int,
    queue: mp.Queue,
    has_writer: bool=False,
    logger=None
):
    """Worker of a training thread."""
    if has_writer:
        writer = SummaryWriter(comment=f"_p{pid}")
    else:
        writer = None

    local_env = deepcopy(env)
    local_agent = A2CAgent(local_env)
    local_agent.set_model(net)
    local_agent.train()
    local_agent.reset()

    n_iter = 0
    total_rewards = []
    steps = deque(maxlen=n_steps)
    batch = []
    while True:
        n_iter += 1
        exp = local_agent.step()
        # logging and early stopping of cycle in the end of episode
        if exp.done:
            reward = local_agent.total_reward
            total_rewards.append(reward)
            m_reward = np.mean(total_rewards[-100:])
            if logger:
                logger.info(
                    f"{pid}: done {len(total_rewards)} games, reward {m_reward:.3f}"
                )
            if writer:
                writer.add_scalar("reward_100", m_reward, n_iter)
                writer.add_scalar("reward", reward, n_iter)

            if m_reward > objective:
                if logger:
                    logger.info(f"Solved in {n_iter} iterations!")
                break

            local_agent.reset()

        # hadling n-steps bellman equation rollback
        steps.append(exp)
        if len(steps) < n_steps and not steps[-1].done:
            continue

        reward = 0.0
        for e in reversed(list(steps)):
            reward *= gamma
            reward += e.reward
        if steps[-1].done:
            last_state = None
        else:
            last_state = steps[-1].state
        batch.append(Experience(steps[0].state, steps[0].action, reward, last_state, steps[0].done))
        if len(steps) == 1:
            steps.clear()

        # accumulating batch
        if len(batch) < batch_size:
            continue

        # batch to tensors
        estates, eactions, erewards, enot_done_idx, elast_states = [], [], [], [], []
        for idx, e in enumerate(batch):
            estates.append(e.state)
            eactions.append(e.action)
            erewards.append(e.reward)
            if e.next_state is not None:
                enot_done_idx.append(idx)
                elast_states.append(e.next_state)

        if local_agent.is_obs_cont:
            states = torch.as_tensor(np.array(estates))
        else:
            states = F.one_hot(torch.as_tensor(estates), local_agent.obs_n)

        if local_agent.is_act_cont:
            actions = torch.as_tensor(np.array(eactions))
        else:
            actions = [int(e) for e in eactions]
            actions = torch.LongTensor(actions)

        # calculating reference values
        rewards = np.array(erewards, dtype=np.float32)
        if enot_done_idx:
            if local_agent.is_obs_cont:
                last_states = torch.as_tensor(np.array(elast_states))
            else:
                last_states = F.one_hot(torch.as_tensor(elast_states), local_agent.obs_n)
            last_vals = net(last_states)[1]
            last_vals_np = last_vals.data.numpy()[:, 0]
            last_vals_np *= gamma ** n_steps
            rewards[enot_done_idx] += last_vals_np
        ref_vals = torch.as_tensor(rewards)
        batch.clear()

        # loss and grad calculation
        net.zero_grad()
        if local_agent.is_act_cont:
            mu, var, values = net(states)
            critic_loss = F.mse_loss(values.squeeze(-1), ref_vals)

            p1 = - ((mu - actions) ** 2) / (2*var.clamp(min=1e-3))
            p2 = - torch.log(torch.sqrt(2 * torch.pi * var))
            log_probs = p1 + p2
            advantages = ref_vals - values.detach()
            actor_loss = -(advantages * log_probs).mean()

            entropy = -(torch.log(2 * torch.pi * var) + 1) / 2
            entropy_loss = entropy_beta * entropy.mean()
        else:
            logits, values = net(states)
            critic_loss = F.mse_loss(values.squeeze(-1), ref_vals)

            log_probs = F.log_softmax(logits, dim=1)
            advantages = ref_vals - values.detach()
            actor_loss = -(advantages * log_probs[range(batch_size), actions]).mean()

            probs = F.softmax(logits, dim=1)
            entropy_loss = entropy_beta * (probs * log_probs).sum(dim=1).mean()
        loss = critic_loss + actor_loss + entropy_loss
        loss.backward()

        # prepare gradients to put them in queue
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
        grads = [
            param.grad.data.numpy()
            if param.grad is not None
            else None
            for param in net.parameters()
        ]
        queue.put(grads)
    queue.put(None)
