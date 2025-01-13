from .stage_executors.B_make_lut import MakeLutExecutor

from .stage_executors.C_find_extrema_serial import FindExtremaSerialExecutor
from .stage_executors.C_find_extrema_parallel import FindExtremaParallelExecutor
from .stage_executors.C_choose_patches import ChoosePatchesExecutor

# After patches are chosen, we can demonstrate loading them
from .stage_executors._0_try_show_patch import TryShowPatchExecutor
from .stage_executors._1_try_show_patch_dist import TryShowPatchDistExecutor

from .stage_executors.D_preprocess_patches_serial import PreprocessPatchesExecutor

# Before training, we can try the dataloader and network
from .stage_executors._E1_train_try_dataloader import TrainingTryDataloaderExecutor
from .stage_executors._E2_train_try_network import TrainingTryNetworkExecutor

from .stage_executors.E_train import TrainingExecutor
from .stage_executors.E_train_no_preprocess import TrainingNoPreprocessExecutor

# Before predicting, we can try the dataloader and model loading
from .stage_executors._F1_predict_test_dataloader import PredictTryDataloaderExecutor
from .stage_executors._F2_predict_test_model_load import PredictTryModelLoadExecutor

from .stage_executors.F_predict import PredictExectutor