This folder contains noise models as they would be generated during the simulation process. Creating them used of 100 noise simulations per detector channel (~330 GB).

These files should be placed in your dataset folder, within a folder titled "NoiseModel"

Alternatively, in the cfg/model/sim/sim.yaml, the noise default can be changed from noise_spatial_corr to noise_variance and anisotropic white noise can be used.