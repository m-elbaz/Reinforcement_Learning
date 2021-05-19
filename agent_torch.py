from collections import deque
import random
import numpy as np
from model_torch import DQN, DuelingDQN
import torch
from fit import fit
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim


class DQNAgent(object):
    """ A simple Deep Q agent """

    def __init__(self, state_size, action_size, model='DQN', lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model == 'DQN':
            self.model = DQN(state_size, action_size).to(self.device)
        else:
            self.model = DuelingDQN(state_size, action_size).to(self.device)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(torch.Tensor(state).to(self.device)).cpu().detach().numpy()
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size=32):
        """ vectorized implementation; 30x speed up compared with for loop """
        minibatch = random.sample(self.memory, batch_size)

        states = np.array([tup[0][0] for tup in minibatch])
        actions = np.array([tup[1] for tup in minibatch])
        rewards = np.array([tup[2] for tup in minibatch])
        next_states = np.array([tup[3][0] for tup in minibatch])
        done = np.array([tup[4] for tup in minibatch])

        # Q(s', a)
        target = rewards + self.gamma * np.amax(
            self.model(torch.Tensor(next_states).to(self.device)).cpu().detach().numpy(), axis=1)
        # end state target is reward itself (no lookahead)
        target[done] = rewards[done]

        # Q(s, a)
        target_f = self.model(torch.Tensor(states).to(self.device)).cpu().detach().numpy()
        # make the agent to approximately map the current state to future discounted reward
        target_f[range(batch_size), actions] = target

        criterion = nn.MSELoss()

        optimizer = optim.Adam(self.model.parameters(), self.lr)

        tensor_x = torch.Tensor(states)  # transform to torch tensor
        tensor_y = torch.Tensor(target_f)

        my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
        data_loader = DataLoader(my_dataset)

        fit(data_loader, self.model, optimizer, criterion, self.device, epoch=1)

        # self.model.fit(states, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
