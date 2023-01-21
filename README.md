# Data Generation

`record.py` can be used to record driving sequences using CARLA. The configuration of the recording
is managed by hydra with the default settings in `configs/config.yaml`. By default, the resulting
recordings are saved in the folder `outputs`.


## Requirements

To produce recordings, you need CARLA 0.9.13 and sufficiently recent version of the pip packages
`carla, hydra-core, numpy, opencv-python, pyyaml, tqdm`.


## Remarks

1. The autopilot is configured to set the vehicle lights. Unfortunately, not all implemented
vehicles have lights. When making recordings at night, remember to exclude all vehicles without
lights (a2, c3, carlacola, cooper_s, grandtourer, leon, mercedes.coupe, wrangler_rubicon, microlino,
micra, patrol, prius, t2).
1. The seed does not work perfectly, at least when driving in different light/weather conditions.
The differences in the recorded frames will grow with the length of the sequence.
1. A useful town-wise data split seems to be: towns 2, 3, 4, 7, 10 for training, towns 1 and 5 for
validation and testing.
1. At least with many vehicles and pedestrians, CARLA seems to be CPU bottlenecked. Therefore it
possible to run multiple recordings on multiple CARLA instances on the same GPU without significant
slowdown.

# Contributors
This repository was developed together with Daniel Siemsen (https://github.com/dsiem) for
projects at the University of Wuppertal.
