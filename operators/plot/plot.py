from typing import Callable, Optional
from dora import DoraStatus

import carla
import random
import numpy as np

import cv2

"""
This operator is not usable during oasis competition, it displays camera output and some data
"""

from dora_utils import (
    get_extrinsic_matrix,
    get_intrinsic_matrix,
    get_projection_matrix,
    local_points_to_camera_view,
    location_to_camera_view,
)


class Operator:
    def __init__(self):
        self.width = 640
        self.height = 640

        self.image = np.zeros((self.width, self.height, 4))
        self.obstacles = []
        self.world_waypoints = []
        self.camera_position = (0, 0, 0)
        self.inv_extrinsic_matrix = None

        self.INTRINSIC_MATRIX = get_intrinsic_matrix(self.width, self.height, 90)

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
        if dora_input["id"] == "camera_position":
            self.camera_position = np.frombuffer(dora_input["data"], np.float32)

            if len(self.camera_position) != 0:
                self.inv_extrinsic_matrix = np.linalg.inv(
                    get_extrinsic_matrix(get_projection_matrix(self.camera_position))
                )

        if dora_input["id"] == "camera_image":
            camera_frame = np.frombuffer(
                dora_input["data"],
                np.uint8,
            ).reshape((self.width, self.height, 4))

            resized_image = camera_frame[:, :, :3]
            self.image = np.ascontiguousarray(resized_image, np.uint8)

        elif dora_input["id"] == "bbox":
            self.obstacles = np.frombuffer(
                dora_input["data"], np.int32
            ).reshape((-1, 6))

        elif dora_input["id"] == "world_waypoints":
            waypoints = np.frombuffer(dora_input["data"], np.float32)
            waypoints = waypoints.reshape((-1, 3))
            waypoints = waypoints[:, :2]

            # Adding z axis for plot

            waypoints = np.hstack(
                (waypoints, -0.5 + np.zeros((waypoints.shape[0], 1)))
            )

            self.world_waypoints = waypoints

        # Draw obstacles

        for obstacle in self.obstacles:
            [min_x, max_x, min_y, max_y, confidence, label] = obstacle

            start = (int(min_x), int(min_y))
            end = (int(max_x), int(max_y))

            cv2.rectangle(self.image, start, end, (0, 255, 0), 2)

        # Draw world_waypoints

        if self.inv_extrinsic_matrix is not None:
            waypoints = location_to_camera_view(
                self.world_waypoints, self.INTRINSIC_MATRIX, self.inv_extrinsic_matrix
            ).T

            waypoints = np.clip(waypoints, 0, 1_000_000)

            for id, waypoint in enumerate(waypoints):
                if np.isnan(waypoint).any():
                    break

                cv2.circle(
                    self.image,
                    (int(waypoint[0]), int(waypoint[1])),
                    3,
                    (
                        int(np.clip(255 - waypoint[2] * 100, 0, 255)),
                        int(np.clip(waypoint[2], 0, 255)),
                        255,
                    ),
                    -1,
                )

        cv2.imshow("image", self.image)
        cv2.waitKey(1)

        return DoraStatus.CONTINUE

    def __del__(self):
        pass
