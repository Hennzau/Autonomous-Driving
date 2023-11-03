import argparse
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

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenario_manager import ScenarioManager
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
from srunner.tools.scenario_parser import ScenarioConfigurationParser

"""
Node that will manage the Agent 'agent.py' and the communication with CARLA Server
"""


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
    CarlaDataProvider.cleanup()

    CarlaDataProvider.cleanup()

    for i, _ in enumerate(ego_vehicles):
        if ego_vehicles[i]:
            if ego_vehicles[i] is not None and ego_vehicles[i].is_alive:
                print("Destroying ego vehicle {}".format(ego_vehicles[i].id))
                ego_vehicles[i].destroy()
            ego_vehicles[i] = None
    ego_vehicles = []


def vehicle_control():
    control = carla.VehicleControl()
    control.steer = 0.0
    control.throttle = 0.5
    control.brake = 0.0
    control.hand_brake = False

    return control


def main():
    # ---------------

    parser = argparse.ArgumentParser(description="Dora-Drives x Scenario Runner",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--timeout', default="10.0",
                        help='Set the CARLA client timeout value in seconds')

    parser.add_argument('--sync', action='store_true',
                        help='Forces the simulation to run synchronously')

    parser.add_argument('--debug', action="store_true", help='Run with debug output')

    parser.add_argument('--scenario', default='FollowLeadingVehicle_1', help='')

    parser.add_argument('--configFile', default='', help='Provide an additional scenario configuration file (*.xml)')

    args = parser.parse_args()

    # --------------

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(float (args.timeout))

    scenario_configurations = ScenarioConfigurationParser.parse_scenario_configuration(
        args.scenario,
        args.configFile)

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

    # Load scenario and run it

    node = Node()

    for event in node:
        if event['id'] == 'tick':
            CarlaDataProvider.on_carla_tick()
            scenario.scenario.scenario_tree.tick_once()

            scenario.ego_vehicles[0].apply_control(vehicle_control())

    scenario.remove_all_actors()

    cleanup(ego_vehicles)


if __name__ == "__main__":
    main()
