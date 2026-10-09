import os
from pathlib import Path
import time
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval

# Override the global default directory
import tempfile
tempfile.tempdir = os.path.join(os.getcwd(), "tmp")

import mmdet3d.models.fusers.cross_modality_consistency

CONFIG = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"

MODEL_OVERRIDES = {
    "pretrain_aux": False,
    "fuser": {
        "type": "CrossModalityConsistencyFuser",
        "in_channels": [80, 256],
        "out_channels": 256,
        "hidden_dim": 256,
        "bias_start": 2.0,
        "alpha_start": 3.0,
        "beta_start": 2.0,
    }
}

def get_run_dir(base_name):
    base_dir = Path("runs") / base_name
    base_dir.mkdir(parents=True, exist_ok=True)
    run_ids = [
        int(p.name) for p in base_dir.iterdir() 
        if p.is_dir() and p.name.isdigit()
    ]
    next_id = max(run_ids) + 1 if run_ids else 1
    next_dir = base_dir / f"{next_id:04d}"
    next_dir.mkdir(exist_ok=False)
    return str(next_dir)

def main():
    dist.init()

    configs.load(CONFIG, recursive=True)
    cfg = Config(recursive_eval(configs), filename=CONFIG)

    cfg.run_dir = get_run_dir("full_cross_modal_consistency")
    cfg.model = {**cfg.model, **MODEL_OVERRIDES}
    cfg.runner.max_epochs = 5
    cfg.checkpoint_config.max_keep_ckpts = 20
    
    # load backbone checkpoints
    cfg.model.encoders.camera.backbone.init_cfg.checkpoint = "pretrained/swint-nuimages-pretrained.pth"
    # cfg.load_from = "pretrained/lidar-only-det.pth"
    cfg.load_from = "runs/cross_modal_consistency/initial_3_epochs/latest.pth"
    
    # strip out radar from the config to avoid FileNotFound errors
    for split in ("train", "val", "test"):
        ds_cfg = cfg.data[split]
        pipe = ds_cfg.get("pipeline") or ds_cfg.dataset.get("pipeline", [])
        ds_cfg.pipeline = [t for t in pipe if t["type"] != "LoadRadarPointsMultiSweeps"]
        for t in ds_cfg.pipeline:
            if t["type"] == "Collect3D" and "radar" in t.get("keys", []):
                t["keys"].remove("radar")
    
    # save the config
    cfg.dump(f"{cfg.run_dir}/configs.yaml")

    logger = get_root_logger(log_file=f"{cfg.run_dir}/train.log")
    logger.info(f"Effective fuser: {cfg.model.fuser}")

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    model = build_model(cfg.model)
    model.init_weights()
    model.train()
    
    # freeze the auxiliary modules
    if hasattr(model, 'fuser') and hasattr(model.fuser, 'freeze_auxiliary'):
        model.fuser.freeze_auxiliary()
        logger.info("Successfully froze auxiliary modules and set to eval mode.")
    else:
        for name, param in model.named_parameters():
            if "camera_ae" in name or "lidar_ae" in name or "predictor" in name:
                param.requires_grad = False
        model.fuser.camera_ae.eval()
        model.fuser.lidar_ae.eval()
        model.fuser.camera_predictor.eval()
        model.fuser.lidar_predictor.eval()
        logger.info("Manually froze auxiliary modules and locked BatchNorm stats.")
    
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    datasets = [build_dataset(cfg.data.train)]
    train_model(model, datasets, cfg, distributed=True, validate=True,
                timestamp=time.strftime("%Y%m%d_%H%M%S"))

if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()