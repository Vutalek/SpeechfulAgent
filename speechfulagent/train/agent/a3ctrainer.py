from collections import deque
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import gymnasium as gym

from .wrappers import RewardWrapper
from speechfulagent.agent import Agent, A2C
from speechfulagent.dataclasses import *


class AgentTrainer:
    def __init__(
        self,
        env: str,
        objective: float,
        gamma: float,
        entropy_beta: float,
        clip_grad: float,
        n_envs: int,
        n_steps: int,
        batch_size: int,
        learning_rate: float,
        logger = None
    ):
        self.env = env
        environ = RewardWrapper(gym.make(env))
        # self.obs_space = environ.observation_space.shape[0]
        self.obs_space = environ.observation_space.n
        self.act_space = environ.action_space.n

        self.objective = objective

        self.gamma = gamma

        self.entropy_beta = entropy_beta
        self.clip_grad = clip_grad

        self.n_envs = n_envs
        self.pool = []

        self.n_steps = n_steps

        self.train_net = A2C(self.obs_space, self.act_space)
        self.optim = optim.Adam(params=self.train_net.parameters(), lr=learning_rate, eps=1e-5)
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.logger = logger

        self.agent = Agent()
        self.agent.net = self.train_net
    
    def train(self) -> Tuple[Agent, EnvInfo, AgentTrainInfo]:
        mp.freeze_support()
        mp.set_start_method("spawn")
        self.agent.net.share_memory()
        q = mp.Queue(maxsize=self.n_envs)
        self.pool = []
        for id in range(self.n_envs):
            process = mp.Process(
                target=worker_function, 
                args=(
                    id,
                    self.env,
                    self.objective,
                    self.gamma,
                    self.n_steps,
                    self.agent.net,
                    self.obs_space,
                    self.entropy_beta,
                    self.clip_grad,
                    q,
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
                if n_iter % self.batch_size == 0:
                    for param, grad in zip(self.agent.net.parameters(), grads_accum):
                        param.grad = torch.FloatTensor(grad)
                    torch.nn.utils.clip_grad_norm_(self.agent.net.parameters(), self.clip_grad)
                    self.optim.step()
                    grads_accum = None
        finally:
            for proc in self.pool:
                proc.terminate()
                proc.join()

        env_info = EnvInfo(
            self.env,
            int(self.obs_space),
            int(self.act_space)
        )
        train_info = AgentTrainInfo(
            n_iter,
            self.objective,
            self.gamma,
            self.batch_size,
            self.n_steps,
            self.learning_rate,
            self.clip_grad,
            self.n_envs
        )
        return self.agent, env_info, train_info
    
def worker_function(
    id: int, 
    env_id: str,
    objective: float,
    gamma: float,
    n_steps: int,
    net: A2C, 
    obs_space: int,
    entropy_beta: float,
    clip_grad: float,
    queue: mp.Queue,
    logger=None
):
    env = RewardWrapper(gym.make(env_id))
    local_agent = Agent()
    local_agent.net = net
    local_agent.reset()
    state, _ = env.reset()
    local_agent.init_state(state)

    total_rewards = deque(maxlen=100)
    batch = []
    counter = 0
    while True:
        exp = local_agent.step(env)
        batch.append(exp)
        counter += 1
        if counter % n_steps == 0 or exp.done:
            if exp.done:
                total_rewards.append(local_agent.total_reward)
                mean_rew = 0.0
                for rew in total_rewards:
                    mean_rew += rew
                mean_rew /= len(total_rewards)
                if logger:
                    logger.info(f"{id}: reward: {total_rewards[-1]:.3f} mean_reward: {mean_rew:.3f}")
                if mean_rew >= objective:
                    break
                local_agent.reset()
                state, _ = env.reset()
                local_agent.init_state(state)
                r = 0
            else:
                state_t = torch.as_tensor(local_agent._ohe(exp.next_state, obs_space))
                # state_t = torch.as_tensor(exp.next_state)
                state_t.unsqueeze_(0)
                _, r = net(state_t)
                r = float(r.item())
            rewards = []
            for i in range(len(batch)-1, -1, -1):
                r = batch[i].reward + r * gamma
                rewards.insert(0, r)
            states, actions = [], []
            for e in batch:
                states.append(e.state)
                # states.append(torch.tensor(e.state))
                actions.append(e.action)
            states_t = F.one_hot(torch.as_tensor(states), obs_space)
            # states_t = torch.stack(states)
            actions_t = torch.as_tensor(actions)
            rewards_t = torch.as_tensor(rewards)
            net.zero_grad()
            logits, values = net(states_t)
            critic_loss = F.mse_loss(values.squeeze(-1), rewards_t)

            log_probs = F.log_softmax(logits, dim=1)
            advantages = rewards_t - values.squeeze(-1).detach()
            actor_loss = -(advantages * log_probs[range(actions_t.size(0)), actions_t]).mean()

            probs = F.softmax(logits, dim=1)
            entropy_loss = entropy_beta * (probs * log_probs).sum(dim=1).mean()

            loss = critic_loss + actor_loss + entropy_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
            grads = [
                param.grad.data.numpy()
                if param.grad is not None
                else None
                for param in net.parameters()
            ]
            queue.put(grads)
            batch.clear()
    queue.put(None)
