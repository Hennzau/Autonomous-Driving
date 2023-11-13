import math

from dora import Node

from carla import VehicleControl

from autoagents.autonomous_agent import AutonomousAgent

import numpy as np

from scipy.spatial.transform import Rotation

import xml.etree.ElementTree as ETree


def get_entry_point():
    return "MyAgent"


class MyAgent(AutonomousAgent):

    def __init__(self, debug=False):
        super().__init__(debug)

        self.node = None

        self.opendrive_map = None
        self.initialization = None  # lat lon ref

        self.destination = (335, 178, 0)

        self.destination_array = np.array([[self.destination[0], self.destination[1], self.destination[2], ]],
                                          np.float32, )

        self.ego_position = None

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
                "width": 640,
                "height": 640,
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

    def to_world_coordinates(self, lat, lon):

        earth_radius = 6378137.0
        scale = math.cos(self.initialization[0] * math.pi / 180.0)

        mx_initial = scale * self.initialization[1] * math.pi * earth_radius / 180.0
        my_initial = (
                scale
                * earth_radius
                * math.log(math.tan((90.0 + self.initialization[0]) * math.pi / 360.0))
        )

        mx = lon / 180.0 * (math.pi * earth_radius * scale)
        my = math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * (
                earth_radius * scale
        )

        x = mx - mx_initial
        y = -(my - my_initial)

        return x, y

    def calculate_ego_position(self, lat, lon, yaw):
        [x, y] = self.to_world_coordinates(lat, lon)

        roll = 0.0
        pitch = 0.0
        [[qx, qy, qz, qw]] = Rotation.from_euler(
            "xyz", [[roll, pitch, yaw]], degrees=False
        ).as_quat()

        return [x, y, 0, qx, qy, qz, qw]

    def calculate_camera_position(self):
        [x, y, z, qx, qy, qz, qw] = self.ego_position

        return [x + float(self.sensors()[0]["x"]) * qx, y + float(self.sensors()[0]["x"]) * qy, z, qx, qy, qz, qw]

    def run_step(self, input_data, timestamp):

        self.node.send_output("ego_world_destination", self.destination_array.tobytes())

        # ----- Retrieve camera data from sensor

        frame_raw_data = input_data["camera"][1]
        camera_frame = frame_raw_data.tobytes()

        self.node.send_output('camera_image', camera_frame)

        # ----- Retrieve Opendrive data from pseudo sensor (only one time)

        if "opendrive_map" in input_data.keys():  # not sure what oasis do, so let's check anyway
            if self.opendrive_map is None:
                opendrive_map = input_data["opendrive_map"][1]["opendrive"]
                self.opendrive_map = opendrive_map

                """
                Convert from waypoints world coordinates to CARLA GPS coordinates
                :return: tuple with lat and lon coordinates
                """

                tree = ETree.ElementTree(ETree.fromstring(self.opendrive_map))

                # default reference
                lat_ref = 42.0
                lon_ref = 2.0

                for opendrive in tree.iter("OpenDRIVE"):
                    for header in opendrive.iter("header"):
                        for georef in header.iter("geoReference"):
                            if georef.text:
                                str_list = georef.text.split(" ")
                                for item in str_list:
                                    if "+lat_0" in item:
                                        lat_ref = float(item.split("=")[1])
                                    if "+lon_0" in item:
                                        lon_ref = float(item.split("=")[1])

                self.initialization = (lat_ref, lon_ref)

                self.node.send_output("opendrive", opendrive_map.encode())

        if "speedometer" in input_data.keys():  # same here
            self.node.send_output("ego_speed", np.array(input_data["speedometer"][1]["speed"], np.float32).tobytes())

        [lat, lon, z] = input_data["GPS"][1]
        yaw = input_data["IMU"][1][-1] - np.pi / 2

        self.ego_position = self.calculate_ego_position(lat, lon, yaw)
        self.node.send_output("ego_position", np.array(self.ego_position, np.float32).tobytes())
        self.node.send_output("camera_position", np.array(self.calculate_camera_position(), np.float32).tobytes())

        return VehicleControl(
            steer=float(0),
            throttle=float(1),
            brake=float(0),
            hand_brake=False,
        )
