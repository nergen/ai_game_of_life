import pygame
from pygame import *


class game_visualization(object):
    """docstring for game_visualization"""
    agent_color = (123, 125, 200)
    background_color = (120, 120, 120)
    food_color = (200, 10, 100)
    game_wait_ms = 1

    def __init__(self, in_width, in_heigh, in_scale):
        if in_scale <= 0:
            raise 'Scale must be greater than 0'
        game_size = (
            in_width * (2 * in_scale + 1),
            in_heigh * (2 * in_scale + 1))
        pygame.init()
        self.scale = in_scale
        self.window = pygame.display.set_mode(game_size)

    def check_event(self, events):
        for event in events:
            if (event.type == QUIT) or \
                (event.type == KEYDOWN and event.key == K_ESCAPE):
                return 0
        return 1

    def drow_map(self, in_map):
        self.window.fill(self.background_color)
        for x, row in enumerate(in_map):
            for y, cell in enumerate(row):
                if len(cell):
                    pygame.draw.circle(
                        self.window,
                        self.agent_color if cell[0] >= 0 else self.food_color,
                        (
                            (2 * self.scale + 1) * x + self.scale,
                            (2 * self.scale + 1) * y + self.scale), self.scale)
        pygame.display.update()
        pygame.time.wait(self.game_wait_ms)