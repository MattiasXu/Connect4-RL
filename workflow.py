import gym
import gym_connect4
import numpy as np
import torch
import torch.nn.functional as F
import sys
import tools
import actors
from mcts import MCTS

rollout = 10
env = gym.make('Connect4Env-v0')
obses = env.reset()  # dict: {0: obs_player_1, 1: obs_player_2}

actor1 = MCTS(rollout, 0, env)
actor2 = actors.RandomActor()

actors = [actor1, actor2]

game_over = False
wins = 0
draws = 0
loses = 0
mcts_idx = 0
for i in range(1000):
    if np.random.rand() < 0.5:
        actors[0], actors[1] = actors[1], actors[0]
        mcts_idx ^= 1

    game_over = False

    obses = env.reset()
    actors[mcts_idx].reset()
    while not game_over:
        action_dict = {}
        for actor_id, actor in enumerate(actors):
            action = actor.act(obses[actor_id])
            actors[actor_id ^ 1].opponent_act(action)
            action_dict[actor_id] = action

        obses, rewards, game_over, info = env.step(action_dict)
        # env.render()
    if rewards[mcts_idx] == 1:
        wins += 1
    elif rewards[mcts_idx] == 0:
        draws += 1
    else:
        loses += 1
    if i % 50 == 49:
        print(f"Games played: {i}")
print(f"W: {wins}, D: {draws}, L: {loses}")
