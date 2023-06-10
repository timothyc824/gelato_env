from typing import Any
from pathlib import Path
import os

import torch


def get_root_dir() -> Path:
    return Path(os.path.dirname(os.path.realpath(__file__))).parent


def first_not_none(*args: Any) -> Any:
    try:
        return next(item for item in args if item is not None)
    except StopIteration:
        raise ValueError(f"All items in list evaluated to None: {list(args)}")


def custom_collate_fn(data):
    public_obs, target = list(zip(*data))
    public_obs_tensor = torch.vstack(public_obs)
    target_tensor = torch.vstack(target)

    return {"public_obs": public_obs_tensor,
            "target": target_tensor}
