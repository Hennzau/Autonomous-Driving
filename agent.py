# Carla IMPORTS

from autoagents.autonomous_agent import AutonomousAgent


class Agent(AutonomousAgent):

    def setup(self, destination, path_to_config_file):
        print("Setup")

    def sensors(self):
        return []

    def run_step(self, input_data, timestamp):


        return VehicleControl(
            steer=float(0),
            throttle=float(0),
            brake=float(0),
            hand_brake=False,
        )
