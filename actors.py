import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomActor():
    """Random actor""" 
    def act(self, obs):
        actions = np.argwhere(obs['action_mask']).reshape(-1)
        return np.random.choice(actions)

class ManualActor():
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
        self.layer2 = nn.Linear(128, 8)
    
    def forward(self, x):
        x = x.view(-1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.softmax(x, dim=-1)
        return x


if __name__ == "__main__":
    

    testboard = torch.rand(6,7,3)
    model = FCPolicy()
    out = model(testboard)
    print(out)