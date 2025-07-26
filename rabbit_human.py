from game.rabbit import RabbitGameAI
import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()



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



game_speed = 5
game_level = 1
if __name__ == '__main__':
    obstacles = obstacles_loc(game_level)
    game = RabbitGameAI(obstacles=obstacles,game_speed=game_speed, game_level=game_level)
    while True:
        _,game_over, score = game.play_step(action=None)
        if game_over:
            break
    print('Final Score:', score)
    pygame.quit()