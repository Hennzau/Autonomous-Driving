from typing import Callable, Optional
from dora import DoraStatus

import carla
import random
import numpy as np

import cv2

"""
This operator is not usable during oasis competition, it displays camera output and some data
"""


class Operator:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(60)
        self.world = self.client.get_world()

        self.width = 640
        self.height = 640

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
        if dora_input["id"] == "image":
            camera_frame = np.frombuffer(
                dora_input["data"],
                np.uint8,
            ).reshape((self.width, self.height, 4))

            resized_image = camera_frame[:, :, :3]
            resized_image = np.ascontiguousarray(resized_image, np.uint8)

            cv2.imshow("image", resized_image)
            cv2.waitKey(1)

        return DoraStatus.CONTINUE

    def __del__(self):
        pass
