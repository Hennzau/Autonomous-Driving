# Carla IMPORTS

from dora import Node

from carla import VehicleControl

from autoagents.autonomous_agent import AutonomousAgent

import numpy as np


class MyAgent(AutonomousAgent):

    def __init__(self, debug=False):
        super().__init__(debug)

        self.node = None

    def setup(self, destination, path_to_config_file):
        self.node = Node()

    def sensors(self):
        return [
            {
                "type": "sensor.camera.rgb",
                "id": "camera",
                "x": 2.0,
                "y": 0.0,
                "z": 1.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 1280,
                "height": 720,
                "fov": 90,
            },
            {
                "type": "sensor.lidar.ray_cast",
                "id": "lidar",
                "x": 2.0,
                "y": 0.0,
                "z": 1.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
            {
                "type": "sensor.other.radar",
                "id": "radar",
                "x": 0.7,
                "y": -0.4,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -45.0,
                "fov": 30
            },
            {
                "type": "sensor.other.gnss",
                "id": "GPS",
                "x": 2,
                "y": 0,
                "z": 1.5,
            },
            {
                "type": "sensor.other.imu",
                "id": "IMU",
                "x": 2,
                "y": 0,
                "z": 1.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
            {
                "type": "sensor.opendrive_map",
                "id": "opendrive_map",
                "reading_frequency": 1,
            },
            {
                "type": "sensor.speedometer",
                "id": "speedometer"
            },
        ]

    def run_step(self, input_data, timestamp):
        frame_raw_data = input_data["camera"][1]
        camera_frame = frame_raw_data.tobytes()

        self.node.send_output('image', camera_frame)

        return VehicleControl(
            steer=float(0),
            throttle=float(1),
            brake=float(0),
            hand_brake=False,
        )
