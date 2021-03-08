import gym
import gym_connect4
import numpy as np
import torch
import torch.nn.functional as F
import sys
import tools
import actors

actor1 = actors.RandomActor()
actor2 = actors.RandomActor()

agents = [actor1, actor2]
env = gym.make('Connect4Env-v0')
obses = env.reset()  # dict: {0: obs_player_1, 1: obs_player_2}


game_over = False
while not game_over:
    action_dict = {}
    for agent_id, agent in enumerate(agents):
        action = agent.act(obses[agent_id])
        action_dict[agent_id] = action
    
    obses, rewards, game_over, info = env.step(action_dict)
    env.render()
    input("Press Enter to continue...")




"""
TODO: Fix one-hot encoding
TODO: Integrate PyTorch
"""