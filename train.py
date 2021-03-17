import gym
import gym_connect4
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import sys
from tools import onehot
import actors
import signal


def compute_loss(actor, obs, act, weights, mask):
    """ Policy Gradient Loss """
    logp = actor.get_policy(obs, mask).log_prob(act)
    return -(logp * weights).mean()

def train_step(env, actors, optimizer, player=0):
    """ Training after playing 1 game """
    game_obs = []
    game_acts = []
    game_mask = []
    game_weights = []

    game_length = 0
    reward = 0

    obses = env.reset()
    game_over = False

    if player == 0:
        training_player_turn = True
    else:
        training_player_turn = False
    repeat = 0
    turns = 0
    while not game_over:
        action_dict = {}
        for actor_id, actor in enumerate(actors):
            obs = obses[actor_id]

            action = actor.act(obs)
            if training_player_turn and actor_id == player:
                game_acts.append(action)
                game_obs.append(onehot(obs['board']))
                game_mask.append(obs['action_mask'])
                game_length += 1
            elif True:
                action = actor.act(obs)

            action_dict[actor_id] = action

        obses, rewards, game_over, info = env.step(action_dict)

        if game_over:
            result = rewards[player]
            reward += rewards[player]
            game_weights = [reward] * game_length

        training_player_turn = not training_player_turn
        turns += 1
    
    optimizer.zero_grad()
    game_loss = compute_loss(actor=actors[player],
                             obs=torch.as_tensor(game_obs, dtype=torch.float32),
                             act=torch.as_tensor(game_acts, dtype=torch.int32),
                             weights=torch.as_tensor(game_weights, dtype=torch.float32),
                             mask=torch.as_tensor(game_mask, dtype=torch.int32).detach(),
                            )

    game_loss.backward()
    optimizer.step()
    return result, game_length


def train():
    def save_model(sig, frame):
        print("Saving model...")
        torch.save(actor_train.state_dict(), "./saved_models/FCPolicyInterrupt.pt")
        sys.exit(0)
    
    env = gym.make('Connect4Env-v0')
    lr = 0.00001
    games = 1000000
    load = True
    save = True

    if save:
        signal.signal(signal.SIGINT, save_model) 

    actor_train = actors.FCPolicy()
    actor_opponent = actors.FCPolicy()
    actor_opponent.load_state_dict(torch.load("./saved_models/FCPolicyInterrupt.pt"))
    actor_opponent.eval()

    train_idx = 0
    actor_list = [actor_train, actor_opponent]

    if load:
        actor_train.load_state_dict(torch.load("./saved_models/FCPolicyInterrupt.pt"))
    optimizer = torch.optim.Adam(actor_train.parameters(), lr=lr)

    wins = 0
    draws = 0
    losses = 0
    game_lengths = 0
    for i in range(games):
        if np.random.rand() < 0.5: # Flip starting player
            train_idx ^= 1
            actor_list[0], actor_list[1] = actor_list[1], actor_list[0]

        result, game_length = train_step(env, actor_list, optimizer, train_idx)
        game_lengths += game_length
        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1
        if i % 100 == 99:
            print(f"Results after 100: {wins}W, {draws}D, {losses}L | Game len {game_lengths}")
            wins = 0
            draws = 0
            losses = 0
            game_lengths = 0
        
        if i % 3000 == 2999:
            print("Checkpoint reached")
            torch.save(actor_train.state_dict(), "./saved_models/checkpoint.pt")
            actor_opponent.load_state_dict(actor_train.state_dict())
    
    torch.save(actor_train.state_dict(), "./saved_models/final.pt")


if __name__ == "__main__":
    train()
    