import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, n_obs, n_action, width=32):
        super(DQN, self).__init__()

        self.input = n_obs
        self.width = width
        self.n_action = n_action

        self.fc1 = nn.Linear(self.input, self.width)
        self.fc2 = nn.Linear(self.width, self.width)
        self.fc3 = nn.Linear(self.width, self.n_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN_LSTM(nn.Module):

    def __init__(self, n_obs, n_action, depth=2):
        super(DQN_LSTM, self).__init__()

        self.input = n_obs
        self.depth = depth
        self.n_action = n_action

        self.rnn = nn.LSTM(self.input, self.n_action, self.depth)


    def forward(self, x):
        return self.rnn(x)

class PG(nn.Module):

    def __init__(self, n_obs, n_action, width=32):
        super(PG, self).__init__()

        self.input = n_obs
        self.width = width
        self.n_action = n_action

        self.fc1 = nn.Linear(self.input, self.width)
        self.fc2 = nn.Linear(self.width, self.width)
        self.fc3 = nn.Linear(self.width, self.n_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=0)

class DuelingDQN(nn.Module):

    def __init__(self, n_obs, n_action, width=32):
        super(DuelingDQN, self).__init__()
        self.input = n_obs
        self.width = width
        self.n_action = n_action

        self.feauture_layer = nn.Sequential(
            nn.Linear(self.input, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.n_action)
        )

    def forward(self, state):
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals

class DuelingDQN_LSTM(nn.Module):

    def __init__(self, n_obs, n_action, depth=2, width=16):
        super(DuelingDQN_LSTM, self).__init__()
        self.input = n_obs
        self.depth = depth
        self.n_action = n_action
        self.width=width

        self.feauture_layer = nn.LSTM(self.n_action, self.width, self.depth+1)

        self.value_stream = nn.LSTM(self.width, 1, self.depth)

        self.advantage_stream = nn.Sequential(self.width, self.n_action, self.depth)


    def forward(self, state):
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals