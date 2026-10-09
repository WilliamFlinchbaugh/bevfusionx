import os
from pathlib import Path
import time
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, recursive_eval, convert_sync_batchnorm

# Override the global default directory
import tempfile
tempfile.tempdir = os.path.join(os.getcwd(), "tmp")

CONFIG = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"

MODEL_OVERRIDES = {
    "pretrain_aux": True,
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

    cfg.run_dir = get_run_dir("cross_modal_consistency")
    cfg.model = {**cfg.model, **MODEL_OVERRIDES}
    cfg.runner.max_epochs = 5
    cfg.checkpoint_config.max_keep_ckpts = 5
    
    # load backbone checkpoints
    cfg.model.encoders.camera.backbone.init_cfg.checkpoint = "pretrained/swint-nuimages-pretrained.pth"
    cfg.load_from = "pretrained/lidar-only-det.pth"
    
    augmentations_to_remove = {
        "ObjectPaste",
        "RandomFlip3D",
        "GridMask",
        "PointShuffle"
    }

    for split in ("train", "val", "test"):
        ds_cfg = cfg.data[split]
        
        if "pipeline" in ds_cfg:
            target_pipeline = ds_cfg.pipeline
        elif "dataset" in ds_cfg and "pipeline" in ds_cfg.dataset:
            target_pipeline = ds_cfg.dataset.pipeline
        else:
            continue
            
        clean_pipeline = []
        for step in target_pipeline:
            if step["type"] == "LoadRadarPointsMultiSweeps":
                continue
                
            if step["type"] == "Collect3D" and "radar" in step.get("keys", []):
                step["keys"].remove("radar")
                
            if split == "train":
                if step["type"] in augmentations_to_remove:
                    continue
                    
                if step["type"] == "GlobalRotScaleTrans":
                    step["rot_lim"] = [0.0, 0.0]
                    step["resize_lim"] = [1.0, 1.0]
                    step["trans_lim"] = 0.0
                
                elif step["type"] == "ImageAug3D":
                    step["rot_lim"] = [0.0, 0.0]
                    step["rand_flip"] = False
                    
            clean_pipeline.append(step)
            
        # Reassign the cleaned pipeline
        if "pipeline" in ds_cfg:
            ds_cfg.pipeline = clean_pipeline
        else:
            ds_cfg.dataset.pipeline = clean_pipeline
    
    # save the config
    cfg.dump(f"{cfg.run_dir}/configs.yaml")

    logger = get_root_logger(log_file=f"{cfg.run_dir}/train.log")
    logger.info(f"Effective fuser: {cfg.model.fuser}")

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    model.eval()
    
    # freeze all parameters except for the auxiliary modules
    for name, param in model.named_parameters():
        if "camera_ae" in name or "lidar_ae" in name or "predictor" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # unfreeze the modules we're training
    if hasattr(model, 'fuser') and hasattr(model.fuser, 'unfreeze_auxiliary'):
        model.fuser.unfreeze_auxiliary()
    
    ds = build_dataset(cfg.data.train)
    
    train_model(
        model,
        ds,
        cfg,
        distributed=True,
        validate=False,
        timestamp=time.strftime("%Y%m%d_%H%M%S", time.localtime())
    )

if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()