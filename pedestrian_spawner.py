import logging
import random

import carla


class PedestrianSpawner:
    def __init__(self, client, num_pedestrians):
        self.client = client
        self.num_pedestrians = num_pedestrians
        self.pedestrian_ids = []
        self.controller_ids = []

    def __enter__(self):
        world = self.client.get_world()
        SpawnActor = carla.command.SpawnActor

        logging.info("Spawning pedestrians...")
        blueprints = world.get_blueprint_library().filter("walker.pedestrian.*")
        batch = [
            SpawnActor(blueprint, carla.Transform(world.get_random_location_from_navigation()))
            for blueprint in random.choices(blueprints, k=self.num_pedestrians)
        ]
        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                logging.error(response.error)
            else:
                self.pedestrian_ids.append(response.actor_id)

        logging.info("Spawning controllers for pedestrians...")
        blueprint = world.get_blueprint_library().find("controller.ai.walker")
        batch = [SpawnActor(blueprint, carla.Transform(), id) for id in self.pedestrian_ids]
        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                logging.error(response.error)
            else:
                self.controller_ids.append(response.actor_id)

        logging.info("Starting controllers for pedestrians...")
        for controller in world.get_actors(self.controller_ids):
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(1 + random.random())

        logging.info(f"Spawned {len(self.pedestrian_ids)} pedestrians.")

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.controller_ids:
            logging.info("Stopping all spawned controllers...")
            for controller in self.client.get_world().get_actors(self.controller_ids):
                controller.stop()

            logging.info("Destroying all spawned actors...")
            ids = self.pedestrian_ids + self.controller_ids
            self.client.apply_batch([carla.command.DestroyActor(id) for id in ids])
