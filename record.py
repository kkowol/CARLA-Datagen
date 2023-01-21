import logging
import random
import multiprocessing
from joblib import Parallel, delayed
from multiprocessing import Array
import carla
import cv2
import hydra
import numpy as np
import omegaconf
import yaml
from tqdm import tqdm

from ego_spawner import EgoSpawner
from pedestrian_spawner import PedestrianSpawner
from sync_mode import SyncMode
from utils import panoptic_segmentation, vector_to_xyz, write_json_bb, write_json_bb_opt
from vehicle_spawner import VehicleSpawner
from weather import Weather
@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg: omegaconf.DictConfig) -> None:
    fences = []
    filenames=[]
    i_strs=[]
    images = []
    measurements_list_height = []
    measurements_list_width = []
    N_cpu = cfg.n_cpu
    if "route_file" in cfg:
        route = omegaconf.OmegaConf.load(hydra.utils.to_absolute_path(cfg.route_file))
    else:
        route = None

    client = carla.Client(cfg.host, cfg.port)
    client.set_timeout(10.0)
    random.seed(cfg.seed)
    bb_image = cfg.bb_img

    try:
        logging.info(f"Loading map {cfg.map}...")
        world = client.load_world(cfg.map)
        wmap = world.get_map()

        if cfg.carla_presets: 
            weather_presets = {
                x: getattr(carla.WeatherParameters, x)
                for x in dir(carla.WeatherParameters)
                if x[0].isupper()
            }
            if cfg.weather in weather_presets:
                logging.info(f"Setting weather {cfg.weather}...")
                world.set_weather(weather_presets[cfg.weather])
            else:
                logging.warning(f"Weather preset {cfg.weather} not found!")
        else:
            weather = Weather(client)
            weather.set_weather(cfg.weather)
            logging.info(f"Setting weather {cfg.weather}...")
        
        world.set_pedestrians_seed(cfg.seed)
        world.set_pedestrians_cross_factor(cfg.pedestrians.cross_factor)

        traffic_manager = client.get_trafficmanager(cfg.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_random_device_seed(cfg.seed)
        traffic_manager.set_synchronous_mode(True)

        logging.info("Removing fences...")
        fences = world.get_environment_objects(carla.CityObjectLabel.Fences)
        world.enable_environment_objects([*map(lambda obj: obj.id, fences)], False)

        spawn_point = None
        if route is not None:
            spawn_point = carla.Transform(
                carla.Location(**route.location), carla.Rotation(**route.rotation)
            )

        if cfg.debug:
            ego = EgoSpawner(
                client,
                cfg.tm_port,
                **cfg.ego_vehicle,
                spawn_point=spawn_point,
                sensors={},
            )
        else:
            ego = EgoSpawner(
                client,
                cfg.tm_port,
                **cfg.ego_vehicle,
                spawn_point=spawn_point,
                sensors=cfg.sensors,
            )

        vehicles = VehicleSpawner(client, cfg.tm_port, cfg.vehicles.num, cfg.vehicles.excluded)
        pedestrians = PedestrianSpawner(client, cfg.pedestrians.num)
        with ego as (vehicle, sensors), vehicles, pedestrians:
            if route is not None:
                traffic_manager.set_route(vehicle, list(route.instructions))

            if cfg.debug:
                traffic_manager.ignore_lights_percentage(vehicle, 100)
            traffic_manager.auto_lane_change(vehicle, False)

            with SyncMode(world, *sensors, fps=cfg.fps) as sync_mode:
                with open("ego.yaml", "w") as file:
                    for i in tqdm(range(cfg.max_frames)):
                        measurements = sync_mode.tick(timeout=1.0)

                        if (i + 1) % cfg.save_every_n_frames == 0: 
                            i_str = f"{i // cfg.save_every_n_frames:0>6}"

                            transform = vehicle.get_transform()
                            wp = wmap.get_waypoint(transform.location)
                            info = {
                                "road_id": wp.road_id,
                                "lane_id": wp.lane_id,
                                "is_junction": wp.is_junction,
                                "location": vector_to_xyz(transform.location),
                                "rotation": {
                                    "pitch": transform.rotation.pitch,
                                    "yaw": transform.rotation.yaw,
                                    "roll": transform.rotation.roll,
                                },
                                "velocity": vector_to_xyz(vehicle.get_velocity()),
                                "acceleration": vector_to_xyz(vehicle.get_acceleration()),
                                "angular_velocity": vector_to_xyz(vehicle.get_angular_velocity()),
                            }
                            yaml.dump({i_str: info}, file)

                            for name, measurement in zip(cfg.sensors, measurements):
                                
                                
                                filename = f"{i_str}_{name}"
                                if isinstance(measurement, carla.Image):
                                    img = np.frombuffer(measurement.raw_data, dtype=np.uint8)
                                    img = img.reshape(measurement.height, measurement.width, 4)
                                    cv2.imwrite(f"{filename}.png", img)
                                else:
                                    measurement.save_to_disk(filename)
                                if name == 'semantic':
                                    measurement.save_to_disk(f'{filename}_cs.png', carla.ColorConverter.CityScapesPalette)
                                if name == 'instance':
                                    filenames.append(filename)
                                    i_strs.append(i_str)
                                    images.append(bb_image)
                                    measurements_list_height.append(measurement.height)
                                    measurements_list_width.append(measurement.width)
                                #     write_json_bb_opt(filename, measurement,  i_str, image)
        Parallel(n_jobs=N_cpu)(delayed(write_json_bb)(filename,measurement_h, measurement_w, i_str, image) for (filename,measurement_h, measurement_w, i_str, image) in zip(filenames,measurements_list_height, measurements_list_width, i_strs, images))                             
        if cfg.panoptic:
            Parallel(n_jobs=N_cpu)(delayed(panoptic_segmentation)(i_str) for i_str in i_strs)

    except KeyboardInterrupt:
        logging.info("Cancelled by user.")

    finally:
        logging.info("Reloading fences...")
        world.enable_environment_objects([*map(lambda obj: obj.id, fences)], True)


if __name__ == "__main__":
    main()