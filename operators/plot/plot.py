from typing import Callable, Optional
from dora import DoraStatus

import carla
import random
import pygame
import numpy as np

"""
This operator is not usable during oasis competition, it displays camera output and some data
"""


class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')

        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))


class Operator:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(60)
        self.world = self.client.get_world()

        self.render_object = RenderObject(1280, 720)

        pygame.init()

        self.game_display = pygame.display.set_mode((1280, 720), pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.game_display.fill((0, 0, 0))
        self.game_display.blit(self.render_object.surface, (0, 0))

        pygame.display.flip()

        print("Plot connected and ready to display something")

    def on_event(
            self,
            dora_event: dict,
            send_output: Callable[[str, bytes, Optional[dict]], None],
    ) -> DoraStatus:
        if dora_event["type"] == "INPUT":
            return self.on_input(dora_event, send_output)
        return DoraStatus.CONTINUE

    def on_input(
            self,
            dora_input: dict,
            send_output: Callable[[str, bytes, Optional[dict]], None],
    ):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return DoraStatus.STOP

        if dora_input["id"] == "image":
            camera_frame = np.frombuffer(
                dora_input["data"],
                np.uint8,
            ).reshape((1280, 720, 4))

            # self.render_object.surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))

        self.game_display.blit(self.render_object.surface, (0, 0))

        pygame.display.flip()

        return DoraStatus.CONTINUE

    def __del__(self):
        pygame.quit()
