import os
import time
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, recursive_eval

# Override the global default directory
import tempfile
tempfile.tempdir = os.path.join(os.getcwd(), "tmp")

import mmdet3d.models.fusers.adaptive

CONFIG = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"

MODEL_OVERRIDES = {}

def main():
    dist.init()

    configs.load(CONFIG, recursive=True)
    cfg = Config(recursive_eval(configs), filename=CONFIG)

    cfg.model = {**cfg.model, **MODEL_OVERRIDES}
    cfg.run_dir = auto_set_run_dir()
    cfg.runner.max_epochs = 30
    
    # strip out radar from the config to avoid FileNotFound errors
    for split in ("train", "val", "test"):
        ds_cfg = cfg.data[split]
        pipe = ds_cfg.get("pipeline") or ds_cfg.dataset.get("pipeline", [])
        ds_cfg.pipeline = [t for t in pipe if t["type"] != "LoadRadarPointsMultiSweeps"]
        for t in ds_cfg.pipeline:
            if t["type"] == "Collect3D" and "radar" in t.get("keys", []):
                t["keys"].remove("radar")
    
    cfg.dump(f"{cfg.run_dir}/configs.yaml")

    logger = get_root_logger(log_file=f"{cfg.run_dir}/train.log")
    logger.info(f"Effective fuser: {cfg.model.fuser}")

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    model = build_model(cfg.model)
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    train_model(model, datasets, cfg, distributed=True, validate=True,
                timestamp=time.strftime("%Y%m%d_%H%M%S"))

if __name__ == "__main__":
    main()