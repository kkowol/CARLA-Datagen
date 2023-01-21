import logging
import random

import carla


class EgoSpawner:
    def __init__(self, client, tm_port, blueprint=None, spawn_point=None, sensors=None):
        self.client = client
        self.world = self.client.get_world()
        self.tm_port = tm_port

        if blueprint is None:
            self.blueprint = random.choice(self.world.get_blueprint_library().filter("vehicle.*"))
        else:
            self.blueprint = self.world.get_blueprint_library().find(blueprint)
        self.blueprint.set_attribute("role_name", "ego")

        if spawn_point is None:
            self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
        else:
            self.spawn_point = spawn_point

        self.sensors = sensors
        self.sensor_actors = []
        self.vehicle = None

    def __enter__(self):
        logging.info("Spawning ego vehicle...")

        self.vehicle = self.world.spawn_actor(self.blueprint, self.spawn_point)
        self.vehicle.set_simulate_physics(False)
        self.vehicle.set_autopilot(True, self.tm_port)

        traffic_manager = self.client.get_trafficmanager(self.tm_port)
        traffic_manager.update_vehicle_lights(self.vehicle, True)

        if self.sensors is not None:
            logging.info("Spawning sensors...")
            for sensor in self.sensors.values():
                blueprint = self.world.get_blueprint_library().find(sensor.blueprint)
                for attribute, value in sensor.attributes.items():
                    blueprint.set_attribute(attribute, str(value))
                location = carla.Location(**sensor.get("location", {}))
                rotation = carla.Rotation(**sensor.get("rotation", {}))
                transform = carla.Transform(location, rotation)
                self.sensor_actors.append(
                    self.world.spawn_actor(blueprint, transform, attach_to=self.vehicle)
                )

        return self.vehicle, self.sensor_actors

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.sensor_actors:
            logging.info("Destroying sensors...")
            for sensor in self.sensor_actors:
                sensor.destroy()

        if self.vehicle is not None:
            logging.info("Destroying ego vehicle...")
            self.vehicle.destroy()
