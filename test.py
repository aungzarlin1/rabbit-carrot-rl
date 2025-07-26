import torch
import pygame
import random
import numpy as np
from collections import deque
# from game import SnakeGameAI, Direction, Point
from game.rabbit import RabbitGameAI, Direction, Point
from model.model import Linear_QNet, QTrainer
from utils.helper import plot
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.rabbit[0]
        point_l = Point(head.x - 30, head.y)
        point_r = Point(head.x + 30, head.y)
        point_u = Point(head.x, head.y - 30)
        point_d = Point(head.x, head.y + 30)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
def obstacles_loc(game_level=2):
    loc = []
    if game_level == 1:
        return loc
    else:
        for _ in range(10):
            x = random.randint(1, 29)
            y = random.randint(1, 19)
            loc.append((x,y))
        return loc


def test(game_level=1):
    agent = Agent()
    agent.model.load_state_dict(torch.load(f"results/model{game_level}_best_score.pth"))
    obstacles = obstacles_loc(game_level)
    game = RabbitGameAI(obstacles=obstacles,game_level=game_level)
    # model.eval()
    # game = RabbitGameAI()
    total_score = 0
    while True:
        state_old = agent.get_state(game)

        # Get action from the trained model (no random exploration)
        final_move = agent.get_action(state_old)

        # Perform the move and get the new state
        reward, done, score = game.play_step(final_move)
        
        # Add the score
        total_score += score

        if done:
            print(f"Game Over! Final Score: {score}")
            game.reset()

        # Optionally, you can render the game state to visually track the performance
        game._update_ui()
        pygame.display.flip()

        if done:
            break

    print(f"Total Score after testing: {total_score}")

if __name__ == "__main__":
    test()
