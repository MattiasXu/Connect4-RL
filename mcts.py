import random
import math

class MCTS():
    def __init__(self, max_rollout, player, env):
        self.max_rollout = max_rollout
        self.player = player ^ 1 # in game class, first player is 1 for some reason
        self.root = Node(None, True, None)
        self.env = env

    def act(self, obs):
        if self.env.game.player != self.player:
            if obs['action_mask'][-1] == 1:
                return 7
            else: # Turn does not match current info
                self.reset_flip()

        game = self.env.game

        for _ in range(self.max_rollout):
            self.rollout(game.clone(), self.root)

        max_visits = 0
        best = []
        for child in self.root.children:
            if child.visits > max_visits:
                best = [child]
                max_visits = child.visits
            elif child.visits == max_visits:
                best.append(child)
        best_child = random.choice(best)
        self.root = best_child
        return best_child.action

    def opponent_act(self, action): # Set new root
        if action == 7:
            return
        for child in self.root.children:
            if child.action == action:
                self.root = child

    def rollout(self, game, root):
        current_node = root
        nodes_to_update = [current_node]

        while not current_node.is_leaf():
            # Traverse tree until leaf
            current_node = current_node.get_best_child()
            game.move(current_node.action)
            nodes_to_update.append(current_node)

        if not game.is_game_over():
            # Expand tree and do random rollout
            current_node.add_children(game.get_moves())
            next_child = current_node.get_best_child()
            nodes_to_update.append(next_child)
            game.move(next_child.action)

            while not game.is_game_over():
                moves = game.get_moves()
                random_move = random.choice(moves)
                game.move(random_move)

        reward = game.get_reward(self.player)

        # Backprop: update nodes
        for node in nodes_to_update:
            node.give_reward(reward)
            node.visits += 1

    def reset(self):
        self.root = Node(None, True, None)

    def flip_player(self):
        self.player = self.player ^ 1

    def reset_flip(self):
        self.reset()
        self.flip_player()

class Node():
    def __init__(self, parent, is_player, action):
        self.parent = parent
        self.is_player = is_player
        self.action = action
        self.children = []
        self.visits = 0
        self.reward = 0

    def is_leaf(self):
        return self.children == []

    def add_children(self, avail_moves):
        for move in avail_moves:
            self.children.append(Node(self, not self.is_player, move))

    def get_best_child(self): # Choose a child with highest UCB
        highest = -math.inf
        best = []
        for child in self.children:
            if child.visits == 0:
                ucb = math.inf
            else:
                ucb = child.reward / child.visits + 2 * (math.log(self.visits)/child.visits) ** 0.5

            if ucb > highest:
                best = [child]
                highest = ucb
            elif ucb == highest:
                best.append(child)
        return random.choice(best)

    def give_reward(self, reward):
        if self.is_player:
            self.reward = reward
        else:
            self.reward = - reward

if __name__ == "__main__":
    test = MCTS(10, 0)
