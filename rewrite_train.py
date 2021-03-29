import gym
import gym_connect4
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from tools import onehot
from actors import *
import signal
from tournament import Tournament

def compute_loss(actor, obs, act, weights, mask):
    logp = actor.model.get_policy(obs, mask).log_prob(act)
    return -(logp * weights).mean()

def evaluate(actors, env):
    tournament = Tournament(actors, 100, env)
    tournament.run()
    return tournament.rewards

def train_step(env, actors, optimizers):
    train_obs = {0: [], 1: []}
    train_action = {0: [], 1: []}
    train_mask = {0: [], 1: []}
    train_weights = {0: [], 1: []}

    game_length = 0

    obses = env.reset()
    player_turn = 0
    game_over = False

    while not game_over:
        action_dict = {}
        for actor_id, actor in enumerate(actors):
            obs = obses[actor_id]
            action = actor.act(obs)
            action_dict[actor_id] = action

            # Record for training
            if actor_id == player_turn:
                train_obs[actor_id].append(onehot(obs['board']))
                train_action[actor_id].append(action)
                train_mask[actor_id].append(obs['action_mask'])
                game_length += 1

        obses, rewards, game_over, info = env.step(action_dict)
        player_turn ^= 1

    for actor_id, actor in enumerate(actors):
        result = rewards[actor_id]
        weights = [result] * len(train_action[actor_id])

        optimizers[actor_id].zero_grad()
        loss = compute_loss(actor=actors[actor_id],
                            obs=torch.as_tensor(train_obs[actor_id], dtype=torch.float32),
                            act=torch.as_tensor(train_action[actor_id], dtype=torch.int32),
                            weights=torch.as_tensor(weights, dtype=torch.float32),
                            mask=torch.as_tensor(train_mask[actor_id], dtype=torch.int32).detach()
                           )
        loss.backward()
        optimizers[actor_id].step()

def train():
    env = gym.make('Connect4Env-v0')

    actor1 = FCActor()
    actor2 = FCActor()

    optimizers = []
    actors = [actor1, actor2]

    for actor in actors:
        optimizers.append(torch.optim.Adam(actor.model.parameters(), lr=0.00001))

    for i in range(100000000):
        train_step(env, actors, optimizers)

        if i % 1000 == 999:
            result = evaluate([actor1, RandomActor()], env)
            print(result)

if __name__ == "__main__":
    train()
