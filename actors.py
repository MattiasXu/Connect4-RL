import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from tools import onehot

class RandomActor():
    """Random actor""" 
    def act(self, obs):
        actions = np.argwhere(obs['action_mask']).reshape(-1)
        return np.random.choice(actions)

class ManualActor():
    """Manual Actor, leave render off in workflow"""
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        actions = np.argwhere(obs['action_mask']).reshape(-1)
        if actions[0] == 7:
            return 7
        else:
            self.env.render()
            action = int(input("Choose column... ")) - 1 
            print(action)
        return action

class FCPolicy(nn.Module):
    """Fully Connected Model"""
    def __init__(self):
        super(FCPolicy, self).__init__()
        self.layer1 = nn.Linear(6*7*3, 128)
        self.layer2 = nn.Linear(128, 7)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.softmax(x, dim=-1)
        return x
    
    def get_policy(self, obs):
        probs = self.forward(obs)
        return Categorical(probs)

    def act(self, obs):
        obs = np.expand_dims(onehot(obs['board']), axis=0)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        return self.get_policy(obs).sample().item()


if __name__ == "__main__":
    

    testboard = torch.rand(6,7,3)
    model = FCPolicy()
    out = model(testboard)
    print(out)