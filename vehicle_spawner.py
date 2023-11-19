import logging
import random
import carla


class VehicleSpawner:
    def __init__(
        self,
        client,
        tm_port,
        num_vehicles,
        excluded=(),
        min_speed_difference=0.0,
        max_speed_difference=30.0,
    ):
        self.client = client
        self.tm_port = tm_port
        self.num_vehicles = num_vehicles
        self.excluded = tuple(excluded)
        self.min_speed_difference = min_speed_difference
        self.max_speed_difference = max_speed_difference
        self.vehicle_ids = []

    def __enter__(self):
        world = self.client.get_world()
        traffic_manager = self.client.get_trafficmanager(self.tm_port)

        FutureActor = carla.command.FutureActor
        SetAutopilot = carla.command.SetAutopilot
        SetSimulatePhysics = carla.command.SetSimulatePhysics

        logging.info("Spawning vehicles...")

        blueprints = [
            blueprint
            for blueprint in world.get_blueprint_library().filter("vehicle.*")
            if (int(blueprint.get_attribute("number_of_wheels")) == 4)
            and not blueprint.id.endswith(self.excluded)
        ]

        batch = []
        for spawn_point in random.sample(world.get_map().get_spawn_points(), k=self.num_vehicles):
            blueprint = random.choice(blueprints)
            blueprint.set_attribute("role_name", "autopilot")
            if blueprint.has_attribute("color"):
                color = random.choice(blueprint.get_attribute("color").recommended_values)
                blueprint.set_attribute("color", color)
            if blueprint.has_attribute("driver_id"):
                driver_id = random.choice(blueprint.get_attribute("driver_id").recommended_values)
                blueprint.set_attribute("driver_id", driver_id)
            spawn_actor = carla.command.SpawnActor(blueprint, spawn_point)
            spawn_actor = spawn_actor.then(SetAutopilot(FutureActor, True, self.tm_port))
            spawn_actor = spawn_actor.then(SetSimulatePhysics(FutureActor, False))
            batch.append(spawn_actor)

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicle_ids.append(response.actor_id)

        vehicles = world.get_actors(self.vehicle_ids)
        for actor in vehicles:
            traffic_manager.vehicle_percentage_speed_difference(
                actor, random.uniform(self.min_speed_difference, self.max_speed_difference)
            )

        logging.info(f"Spawned {len(self.vehicle_ids)} vehicles.")

        return vehicles

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.vehicle_ids:
            logging.info("Destroying spawned vehicles...")
            self.client.apply_batch([carla.command.DestroyActor(id) for id in self.vehicle_ids])
