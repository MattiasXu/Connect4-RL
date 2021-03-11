import gym
import gym_connect4
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import sys
from tools import onehot
import actors

def compute_loss(actor, obs, act, weights):
    """ Policy Gradient Loss """
    logp = actor.get_policy(obs).log_prob(act)
    return -(logp * weights).mean()

def train_step(env, actors, optimizer, player=0):
    """ Training after playing 1 game """
    game_obs = []
    game_acts = []
    game_weights = []

    game_length = 0
    reward = None

    obses = env.reset()
    game_over = False

    if player == 0:
        training_player_turn = True
    else:
        training_player_turn = False

    turns = 0
    while True:
        action_dict = {}
        for actor_id, actor in enumerate(actors):
            obs = obses[actor_id]
            action = actor.act(obs)

            if training_player_turn and actor_id == 0:
                while True: # Keep sampling actions until it is valid
                    game_acts.append(np.expand_dims(action, axis=0))
                    game_obs.append(onehot(obs['board']))
                    game_length += 1
                    
                    if env.game.is_valid_move(action):
                        break
                    action = actor.act(obs)
            
            action_dict[actor_id] = action

        obses, rewards, game_over, info = env.step(action_dict)

        if game_over:
            reward = rewards[0]
            game_weights = [reward] * game_length
            break

        training_player_turn = not training_player_turn
        turns += 1
    
    optimizer.zero_grad()
    game_loss = compute_loss(actor=actors[0],
                             obs=torch.as_tensor(game_obs, dtype=torch.float32),
                             act=torch.as_tensor(game_acts, dtype=torch.int32),
                             weights=torch.as_tensor(game_weights, dtype=torch.float32)
                            )

    game_loss.backward()
    optimizer.step()
    return reward


def train():
    env = gym.make('Connect4Env-v0')

    lr = 0.001
    games = 100000

    actor1 = actors.FCPolicy()
    actor2 = actors.RandomActor()
    actor_list = [actor1, actor2]

    optimizer = torch.optim.Adam(actor1.parameters(), lr=lr)

    reward = 0
    for i in range(games):
        reward += train_step(env, actor_list, optimizer)
        if i % 100 == 99:
            print(f"Avg Reward after 100: {reward}")
            reward = 0

if __name__ == "__main__":
    train()