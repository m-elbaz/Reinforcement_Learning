from collections import deque
import random
import numpy as np
from model_torch import PG
import torch
from torch.distributions import Categorical
from fit import fit
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim


class PGAgent(object):

    def __init__(self, state_size, action_size, model='PG', lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = PG(state_size, action_size).to(self.device)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_prob= self.model(torch.Tensor(state).to(self.device))
        distrib = torch.multinomial(act_prob,1).cpu().detach().numpy()
        return distrib[0][0]  # returns action

    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, batch_size)

        states = np.array([tup[0][0] for tup in minibatch])
        actions = np.array([tup[1] for tup in minibatch])
        rewards = np.array([tup[2] for tup in minibatch])
        next_states = np.array([tup[3][0] for tup in minibatch])
        done = np.array([tup[4] for tup in minibatch])

        # Q(s', a)
        discounted_rewards = np.zeros_like(rewards)
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = cumulative * self.gamma + rewards[t]
            discounted_rewards[t] = cumulative

        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        optimizer = optim.Adam(self.model.parameters(), self.lr)
        optimizer.zero_grad()

        for i in range(batch_size):
            state = states[i]
            action = torch.autograd.Variable(torch.FloatTensor([actions[i]]))
            reward = discounted_rewards[i]

            probs = self.model(torch.Tensor(state).to(self.device))
            m = Categorical(probs)
            loss = -m.log_prob(action) * reward  # Negtive score function x reward
            loss.backward()

        optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
