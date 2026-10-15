"""
SeinePINN_1D: a hybrid LSTM physics-informed neural network (PINN) for the
1-D Saint-Venant equations of a tidal river (stage, velocity and discharge).

A river is described by a YAML case file (see ``cases/``) and run with::

    python run_train.py cases/seine.yaml
    python run_evaluate.py cases/seine.yaml --period test
    python run_predict.py cases/seine.yaml --start 2018-01-22 --end 2018-02-08

Nothing in the package is specific to one river: chainage, datum, column
names, tributaries, tidal constituents, roughness bounds and the physical
scales come from the case file or from the case's own data.
"""

from .config import (CaseConfig, PhysicsConfig, TrainConfig, ModelConfig,
                     DEVICE, G)
from .data import RiverDataset, Window
from .geometry import (CrossSectionLibrary, RiverGeometry, GeometrySurrogate,
                       build_geometry, pretrain_surrogate, section_properties)
from .models import RiverPINN, Context
from .scales import Scales, derive_scales
from .engine import (Case, train, evaluate, predict_grid,
                     save_checkpoint, load_checkpoint)

__version__ = "1.0.0"

__all__ = [
    "CaseConfig", "PhysicsConfig", "TrainConfig", "ModelConfig",
    "DEVICE", "G",
    "RiverDataset", "Window",
    "CrossSectionLibrary", "RiverGeometry", "GeometrySurrogate",
    "build_geometry", "pretrain_surrogate", "section_properties",
    "RiverPINN", "Context", "Scales", "derive_scales",
    "Case", "train", "evaluate", "predict_grid",
    "save_checkpoint", "load_checkpoint",
]
