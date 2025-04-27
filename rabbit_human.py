from game.rabbit import RabbitGameAI
import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()


game_speed = 5
game_level = 2
if __name__ == '__main__':
    game = RabbitGameAI(game_speed=game_speed, game_level=game_level)
    while True:
        _,game_over, score = game.play_step(action=None)
        if game_over:
            break
    print('Final Score:', score)
    pygame.quit()