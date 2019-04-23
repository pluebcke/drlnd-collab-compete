import torch
import torch.nn as nn
import torch.nn.functional as functional


class Actor(nn.Module):
    def __init__(self, state_size, action_size, settings):
        super(Actor, self).__init__()

        self.BN1 = nn.BatchNorm1d(state_size)
        self.FC1 = nn.Linear(state_size, settings['layer1_size'])
        self.FC2 = nn.Linear(settings['layer1_size'], settings['layer2_size'])
        self.FC3 = nn.Linear(settings['layer2_size'], action_size)
        self.reset(settings['out_noise'])

    def forward(self, x):
        x = self.BN1(x)
        x = functional.leaky_relu(self.FC1(x))
        x = functional.leaky_relu(self.FC2(x))
        return functional.tanh(self.FC3(x))

    def get_action(self, x):
        self.eval()
        x = self.BN1(x)
        x = functional.leaky_relu(self.FC1(x))
        x = functional.leaky_relu(self.FC2(x))
        x = functional.tanh(self.FC3(x))
        self.train()
        return x

    def reset(self, out_noise):
        torch.nn.init.xavier_uniform_(self.FC1.weight.data)
        torch.nn.init.xavier_uniform_(self.FC2.weight.data)
        torch.nn.init.uniform_(self.FC3.weight.data, a=-out_noise, b=out_noise)

class Critic(nn.Module):
    # The idea to include both neural networks into one class is originally from the TD3 implementation from
    # https://github.com/sfujim/TD3/
    def __init__(self, state_size, action_size, settings):
        super(Critic, self).__init__()
        self.FC1A = nn.Linear(state_size + action_size, settings['layer1_size'])
        self.FC2A = nn.Linear(settings['layer1_size'], settings['layer2_size'])
        self.FC3A = nn.Linear(settings['layer2_size'], 1)

        self.FC1B = nn.Linear(state_size + action_size, settings['layer1_size'])
        self.FC2B = nn.Linear(settings['layer1_size'], settings['layer2_size'])
        self.FC3B = nn.Linear(settings['layer2_size'], 1)
        self.reset(settings['out_noise'])

    def forward(self, x, action):
        qa = self.get_qa(x, action)
        qb = self.get_qb(x, action)
        return qa, qb

    def get_qa(self, x, action):
        xa = torch.cat([x, action], dim=1)
        xa = functional.leaky_relu(self.FC1A(xa))
        xa = functional.leaky_relu(self.FC2A(xa))
        return self.FC3A(xa)

    def get_qb(self, x, action):
        xb = torch.cat([x, action], dim=1)
        xb = functional.leaky_relu(self.FC1B(xb))
        xb = functional.leaky_relu(self.FC2B(xb))
        return self.FC3B(xb)

    def reset(self, out_noise):
        torch.nn.init.xavier_uniform_(self.FC1A.weight.data)
        torch.nn.init.xavier_uniform_(self.FC2A.weight.data)
        torch.nn.init.uniform_(self.FC3A.weight.data, a=-out_noise, b=out_noise)

        torch.nn.init.xavier_uniform_(self.FC1B.weight.data)
        torch.nn.init.xavier_uniform_(self.FC2B.weight.data)
        torch.nn.init.uniform_(self.FC3B.weight.data, a=-out_noise, b=out_noise)
