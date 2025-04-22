import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 30
# SPEED = 10
PROB_WALL = 0.01
# GAME_LEVEL = 1
MAX_ENEMIES = 2

class RabbitGameAI:

    def __init__(self, w=30 * BLOCK_SIZE, h=20 * BLOCK_SIZE, game_speed = 1000, game_level = 1):
        self.w = w
        self.h = h
        self.game_speed = game_speed
        self.game_level = game_level
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Rabbit Game')
        self.clock = pygame.time.Clock()
        self.reset()

        self.rabbit_run_imgs = [
            pygame.transform.scale(pygame.image.load(f'game/rabbit_run{i}.png').convert_alpha(), (BLOCK_SIZE, BLOCK_SIZE))
            for i in range(1, 3)
        ]
        self.animation_index = 0
        self.animation_timer = 0
        self.animation_delay = 2

        self.carrot_img = pygame.image.load('game/carrot.png').convert_alpha()
        self.carrot_img = pygame.transform.scale(self.carrot_img, (BLOCK_SIZE, BLOCK_SIZE))

        self.wall_img = pygame.image.load('game/wall.png').convert_alpha()
        self.wall_img = pygame.transform.scale(self.wall_img, (BLOCK_SIZE, BLOCK_SIZE))

        if self.game_level > 2:
            self.fox_imgs = [pygame.transform.scale(pygame.image.load(f'game/fox{i}.png').convert_alpha(), (BLOCK_SIZE, BLOCK_SIZE)) for i in range(1, 3)]

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.rabbit = [self.head]
        self.score = 0
        self.wall = self._generate_maze()
        self.food = None
        self._place_carrot()
        self.frame_iteration = 0 
        self.enemies = []
        if self.game_level > 2:
            for _ in range(MAX_ENEMIES):
                enemy = Point(random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
                              random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE)
                enemy_direction = random.choice(list(Direction))
                self.enemies.append({'position': enemy, 'direction': enemy_direction})

    def _generate_maze(self):
        maze = [[0 for _ in range(self.w // BLOCK_SIZE)] for _ in range(self.h // BLOCK_SIZE)]
        for x in range(self.w // BLOCK_SIZE):
            maze[0][x] = 1
            maze[self.h // BLOCK_SIZE - 1][x] = 1
        for y in range(self.h // BLOCK_SIZE):
            maze[y][0] = 1
            maze[y][self.w // BLOCK_SIZE - 1] = 1
        for y in range(1, self.h // BLOCK_SIZE - 1):
            for x in range(1, self.w // BLOCK_SIZE - 1):
                if random.random() < PROB_WALL and self.game_level > 1:
                    maze[y][x] = 1
        wall_points = [Point(x * BLOCK_SIZE, y * BLOCK_SIZE) for y in range(self.h // BLOCK_SIZE)
                       for x in range(self.w // BLOCK_SIZE) if maze[y][x] == 1]
        return wall_points

    def _place_carrot(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.rabbit or self.food in self.wall:
            self._place_carrot()

    def play_step(self, action):
        self.frame_iteration  += 1
        if action == None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                        self.direction = Direction.LEFT
                    elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                        self.direction = Direction.RIGHT
                    elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                        self.direction = Direction.UP
                    elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                        self.direction = Direction.DOWN
            self._move(self.direction, who='human')

        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            self._move(action) # update the head

        self.rabbit.insert(0, self.head)
        self.rabbit.pop()


        # 3. check if game over
        reward = 0
        game_over = False

        if self.is_collision() or self.frame_iteration > 100*(self.score+1) or self.score > 200:
        # if self.is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_carrot()
        # else:
            # reward = -1
        else:
            reward = -0.1
        # self.rabbit.pop()

        for enemy in self.enemies:
            self._move_enemy(enemy)

        self._update_ui()
        self.clock.tick(self.game_speed)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        # if pt in self.rabbit[1:]:
        #     return True
        if pt in self.wall:
            return True
        if pt in [enemy['position'] for enemy in self.enemies]:
            return True
        return False

    def _move_enemy(self, enemy):
        x, y = enemy['position'].x, enemy['position'].y
        if enemy['direction'] == Direction.RIGHT:
            x += BLOCK_SIZE
        elif enemy['direction'] == Direction.LEFT:
            x -= BLOCK_SIZE
        elif enemy['direction'] == Direction.DOWN:
            y += BLOCK_SIZE
        elif enemy['direction'] == Direction.UP:
            y -= BLOCK_SIZE

        if self._is_collision(Point(x, y)):
            enemy['direction'] = random.choice(list(Direction))
        else:
            enemy['position'] = Point(x, y)

    def _update_ui(self):
        self.display.fill(BLACK)
        self.animation_timer += 1
        if self.animation_timer >= self.animation_delay:
            self.animation_timer = 0
            self.animation_index = (self.animation_index + 1) % len(self.rabbit_run_imgs)

        for pt in self.rabbit:
            frame = self.rabbit_run_imgs[self.animation_index]
            self.display.blit(frame, (pt.x, pt.y))

        for w in self.wall:
            self.display.blit(self.wall_img, (w.x, w.y))

        self.display.blit(self.carrot_img, (self.food.x, self.food.y))

        for enemy in self.enemies:
            fox_frame = self.fox_imgs[self.animation_index % 2]
            self.display.blit(fox_frame, (enemy['position'].x, enemy['position'].y))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [10, 10])
        pygame.display.flip()

    def _move(self, action, who='RL'):
        if who == 'RL':
            clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = clockwise.index(self.direction)

            if np.array_equal(action, [1, 0, 0]):
                new_dir = clockwise[idx]
            elif np.array_equal(action, [0, 1, 0]):
                new_dir = clockwise[(idx + 1) % 4]
            else:
                new_dir = clockwise[(idx - 1) % 4]

            self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
