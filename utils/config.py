from dataclasses import dataclass, field
from typing import Iterable, Union, Optional, List, Dict, Sequence, Callable, Any, TypeVar

import torch

from pathlib import Path

from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import Callback
from torch.utils.data import Sampler

from data_generators.gaussian_generators import StrongSeasonalGaussian
from data_generators.generator import Generator
from data_generators.sigmoid_generators import SigmoidGaussian
from env.gelateria import GelateriaState, default_init_state
from utils.misc import get_root_dir, custom_collate_fn

ROOT_DIR = get_root_dir()

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


@dataclass
class BaseConfig:

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class NetConfig(BaseConfig):
    input_key: str = "public_obs"
    target: str = "target"
    embedding_dims: List[int] = field(default_factory=lambda: [32, 32, 32])
    activation: str = "ReLU"
    lr: float = 1e-3
    lr_scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_config: Optional[Dict] = None
    optim: torch.optim = torch.optim.Adam
    path_to_model: Path = ROOT_DIR / "experiment_data/trained_models"


@dataclass
class LightningConfig(BaseConfig):
    default_root_dir: Optional[str] = ROOT_DIR / "experiment_data/training_logs"
    callbacks: Optional[Union[List[Callback], Callback]] = None
    enable_progress_bar: bool = True
    track_grad_norm: Union[int, float, str] = 2.
    max_epochs: Optional[int] = 10
    accelerator: Optional[Union[str, Accelerator]] = "auto"
    auto_lr_find: Union[bool, str] = True
    auto_scale_batch_size: Union[str, bool] = True


@dataclass
class DataGenerationConfig(BaseConfig):
    dir_path: Path = ROOT_DIR / "experiment_data"
    data_filename: str = "experiment_data.csv"
    target_name: str = "sales"
    time_period_in_days: int = 365
    expected_sales_generator: Generator = StrongSeasonalGaussian()
    uplift_generator: Generator = SigmoidGaussian()
    cache_data: bool = True
    init_state: GelateriaState = default_init_state()


@dataclass
class DataLoaderConfig(BaseConfig):
    batch_size: Optional[int] = 64
    train_val_split: float = 0.8
    shuffle: Optional[bool] = True
    sampler: Union[Sampler, Iterable, None] = None
    batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None
    collate_fn: Optional[Callable[[List[T]], Any]] = custom_collate_fn
    pin_memory: bool = False
    drop_last: bool = False
    timeout: float = 0
    worker_init_fn: Optional[Callable[[int], None]] = None
    multiprocessing_context = None
    generator = None
    prefetch_factor: int = 2
    persistent_workers: bool = False
    pin_memory_device: str = ""


@dataclass
class OptimiserConfig(BaseConfig):
    pass


@dataclass
class ExperimentConfig(BaseConfig):
    net_config: NetConfig = NetConfig()
    lightning_config: LightningConfig = LightningConfig()
    data_generation_config: DataGenerationConfig = DataGenerationConfig()
    dataloader_config: DataLoaderConfig = DataLoaderConfig()
    optimiser_config: Optional[OptimiserConfig] = None