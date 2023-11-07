import glob
import importlib
import inspect
import os
import sys
import time
import traceback

import py_trees
from dora import Node

import carla

import copy
import numpy as np

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenario_manager import ScenarioManager
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
from srunner.tools.scenario_parser import ScenarioConfigurationParser

from scipy.spatial.transform import Rotation

"""
Node that will manage the Agent 'my_agent.py' and the communication with CARLA Server
"""

from my_agent import MyAgent


def get_scenario_class_or_fail(scenario):
    """
    Get scenario class by scenario name
    If scenario is not supported or not found, exit script
    """

    # Path of all scenario at "srunner/scenarios" folder + the path of the additional scenario argument
    scenarios_list = glob.glob("srunner/scenarios/*.py")

    for scenario_file in scenarios_list:

        # Get their module
        module_name = os.path.basename(scenario_file).split('.')[0]
        sys.path.insert(0, os.path.dirname(scenario_file))
        scenario_module = importlib.import_module(module_name)

        # And their members of type class
        for member in inspect.getmembers(scenario_module, inspect.isclass):
            if scenario in member:
                return member[1]

        # Remove unused Python paths
        sys.path.pop(0)

    print("Scenario '{}' not supported ... Exiting".format(scenario))
    sys.exit(-1)


def cleanup(ego_vehicles):
    for sensor in sensors:
        sensor.stop()
        sensor.destroy()

    CarlaDataProvider.cleanup()

    for i, _ in enumerate(ego_vehicles):
        if ego_vehicles[i]:
            if ego_vehicles[i] is not None and ego_vehicles[i].is_alive:
                print("Destroying ego vehicle {}".format(ego_vehicles[i].id))
                ego_vehicles[i].destroy()
            ego_vehicles[i] = None
    ego_vehicles = []


# -----------

# --- sensors

# couple (id , data [buffer])
camera = ()
lidar = ()
gnss = ()
imu = ()
radar = ()

sensors = []


def on_camera(data):
    global camera

    camera = ('camera', np.frombuffer(data.raw_data, np.uint8))


def on_lidar(data):
    global lidar

    frame = np.frombuffer(data.raw_data, np.float32)
    point_cloud = np.reshape(frame, (-1, 4))
    point_cloud = point_cloud[:, :3]

    lidar = ('lidar', point_cloud)


def on_gnss(data):
    global gnss

    array = np.array([data.latitude,
                      data.longitude,
                      data.altitude], dtype=np.float64)

    gnss = ('GPS', array)


def on_imu(data):
    global imu

    array = np.array([data.accelerometer.x,
                      data.accelerometer.y,
                      data.accelerometer.z,
                      data.gyroscope.x,
                      data.gyroscope.y,
                      data.gyroscope.z,
                      data.compass,
                      ], dtype=np.float64)

    imu = ('IMU', array)


def on_radar(data):
    global radar

    points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
    points = copy.deepcopy(points)
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    points = np.flip(points, 1)

    radar = ('radar', points)


# -----------

def setup_sensors(agent, vehicle):
    global sensors
    bp_library = CarlaDataProvider.get_world().get_blueprint_library()

    for sensor_spec in agent.sensors():

        bp = None
        sensor_location = None
        sensor_rotation = None

        if sensor_spec['type'].startswith('sensor.camera'):
            bp = bp_library.find(str(sensor_spec['type']))
            bp.set_attribute('image_size_x', str(sensor_spec['width']))
            bp.set_attribute('image_size_y', str(sensor_spec['height']))
            bp.set_attribute('fov', str(sensor_spec['fov']))

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])

            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif sensor_spec['type'].startswith('sensor.lidar.ray_cast'):
            bp = bp_library.find(str(sensor_spec['type']))
            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])

            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif sensor_spec['type'].startswith('sensor.other.gnss'):
            bp = bp_library.find(str(sensor_spec['type']))

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])

            sensor_rotation = carla.Rotation()

        elif sensor_spec['type'].startswith('sensor.other.imu'):
            bp = bp_library.find(str(sensor_spec['type']))

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])

            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif sensor_spec['type'].startswith('sensor.other.radar'):
            bp = bp_library.find(str(sensor_spec['type']))

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])

            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])
        else:
            break

        # create sensor
        sensor_transform = carla.Transform(sensor_location, sensor_rotation)
        sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, vehicle)
        sensors.append(sensor)

        # setup callback

        if sensor_spec['type'].startswith('sensor.camera'):
            sensor.listen(on_camera)

        elif sensor_spec['type'].startswith('sensor.lidar.ray_cast'):
            sensor.listen(on_lidar)

        elif sensor_spec['type'].startswith('sensor.other.gnss'):
            sensor.listen(on_gnss)

        elif sensor_spec['type'].startswith('sensor.other.imu'):
            sensor.listen(on_imu)

        elif sensor_spec['type'].startswith('sensor.other.radar'):
            sensor.listen(on_radar)

        # Tick once to spawn the sensors
        CarlaDataProvider.get_world().tick()


def main():
    # --------------

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(20.0)

    scenario_configurations = ScenarioConfigurationParser.parse_scenario_configuration(
        'FollowLeadingVehicle_1',
        '')

    config = scenario_configurations[0]

    world = client.load_world(config.town)
    world = client.get_world()

    CarlaDataProvider.set_client(client)
    CarlaDataProvider.set_world(world)

    if CarlaDataProvider.is_sync_mode():
        world.tick()
    else:
        world.wait_for_tick()

    CarlaDataProvider.set_traffic_manager_port(8000)
    tm = client.get_trafficmanager(8000)
    tm.set_random_device_seed(0)

    ego_vehicles = []

    print("Preparing scenario: " + config.name)

    # ---- prepare ego vehicles

    for vehicle in config.ego_vehicles:
        ego_vehicles.append(CarlaDataProvider.request_new_actor(vehicle.model,
                                                                vehicle.transform,
                                                                vehicle.rolename,
                                                                color=vehicle.color,
                                                                actor_category=vehicle.category))
    if CarlaDataProvider.is_sync_mode():
        world.tick()
    else:
        world.wait_for_tick()

    # ----

    scenario_class = get_scenario_class_or_fail(config.type)
    scenario = scenario_class(world,
                              ego_vehicles,
                              config,
                              False,
                              False)

    # load personalized agent 'my_agent' (initialize sensors etc...)

    my_agent = MyAgent()
    my_agent.setup(None, None)

    setup_sensors(my_agent, ego_vehicles[0])

    position = None
    last_position = None

    # run scenario and my_agent.py

    for event in my_agent.node:
        if event['id'] == 'tick':
            CarlaDataProvider.on_carla_tick()
            scenario.scenario.scenario_tree.tick_once()

            # manage data to provide to agent

            # calculate speed

            vec_transform = ego_vehicles[0].get_transform()

            x = vec_transform.location.x
            y = vec_transform.location.y
            z = vec_transform.location.z

            yaw = vec_transform.rotation.yaw
            pitch = vec_transform.rotation.pitch
            roll = vec_transform.rotation.roll

            [[qx, qy, qz, qw]] = Rotation.from_euler(
                "xyz", [[roll, pitch, yaw]], degrees=True
            ).as_quat()

            position = np.array([x, y, z, qx, qy, qz, qw], np.float32)
            if last_position is None:
                last_position = position

            speed = np.array([position[:2] - last_position[:2]], np.float32)

            input_data = {
                camera[0]: (0, camera[1]),
                lidar[0]: (0, lidar[1]),
                gnss[0]: (0, gnss[1]),
                imu[0]: (0, imu[1]),
                radar[0]: (0, radar[1]),
                'opendrive_map': (0, {"opendrive": world.get_map().to_opendrive()}),
                'speedometer': (0, {"speed": speed})
            }

            scenario.ego_vehicles[0].apply_control(my_agent.run_step(input_data, 0))

            last_position = position

    scenario.remove_all_actors()

    cleanup(ego_vehicles)


if __name__ == "__main__":
    main()
