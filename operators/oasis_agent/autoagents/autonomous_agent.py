from __future__ import print_function

import carla


class AutonomousAgent(object):

    def __init__(self, debug=False):
        pass

    def setup(self, destination, path_to_conf_file):
        pass

    def sensors(self):  # pylint: disable=no-self-use
        return []

    def run_step(self, input_data, timestamp):
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control

    def destroy(self):
        pass
