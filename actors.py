import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from tools import onehot
from neural_networks import FCModel

class Actor():
    # Mostly to make all actors compatible with MCTS
    def opponent_act(self, action):
        return
    def reset(self):
        return

class RandomActor(Actor):
    """Random actor"""
    def act(self, obs):
        actions = np.argwhere(obs['action_mask']).reshape(-1)
        return np.random.choice(actions)

class ManualActor(Actor):
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
        return action

class FCActor(Actor):
    def __init__(self):
        self.model = FCModel()

    def act(self, obs):
        return self.model.act(obs)

if __name__ == "__main__":


    testboard = torch.rand(6,7,3)
    model = FCPolicy()
    out = model(testboard)
    print(out)
