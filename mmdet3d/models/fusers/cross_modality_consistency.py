from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["CrossModalityConsistencyFuser"]

class SelfVerificationAutoEncoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        C = in_channels
        # Lidar: 256->128->64->128->256
        # Camera: 80->40->20->40->80
        self.encoder = nn.Sequential(
            nn.Conv2d(C, C // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(C // 2),
            nn.GELU(),
            nn.Conv2d(C // 2, C // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C // 4),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'), 
            nn.Conv2d(C // 4, C // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C // 2),
            nn.GELU(),
            nn.Conv2d(C // 2, C, kernel_size=1, stride=1)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        mse = F.mse_loss(x, recon, reduction='none').mean(dim=1, keepdim=True)
        return mse

class CrossModalPredictor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int):
        super().__init__()
        
        # ConvNeXt style inverted bottleneck
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3, groups=hidden_dim),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim*2, out_channels, kernel_size=1),
        )
    
    def forward(self, input_features, target_features):
        predicted = self.predictor(input_features)
        mse = F.mse_loss(predicted, target_features.detach(), reduction='none').mean(dim=1, keepdim=True)
        return mse


@FUSERS.register_module()
class CrossModalityConsistencyFuser(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        hidden_dim: int = 256,
        bias_start: float = 2.0,
        alpha_start: float = 3.0,
        beta_start: float = 2.0,
        use_self_verification: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.bias_cam = nn.Parameter(torch.tensor(bias_start, dtype=torch.float32))
        self.alpha_cam = nn.Parameter(torch.tensor(alpha_start, dtype=torch.float32))
        self.beta_cam = nn.Parameter(torch.tensor(beta_start, dtype=torch.float32))
        
        self.bias_lidar = nn.Parameter(torch.tensor(bias_start, dtype=torch.float32))
        self.alpha_lidar = nn.Parameter(torch.tensor(alpha_start, dtype=torch.float32))
        self.beta_lidar = nn.Parameter(torch.tensor(beta_start, dtype=torch.float32))
        
        camera_channels = in_channels[0]
        lidar_channels = in_channels[1]
        
        self.camera_ae = SelfVerificationAutoEncoder(in_channels=camera_channels)
        self.lidar_ae = SelfVerificationAutoEncoder(in_channels=lidar_channels)
        
        self.camera_predictor = CrossModalPredictor(
            in_channels=camera_channels,
            out_channels=lidar_channels,
            hidden_dim=hidden_dim
        )
        self.lidar_predictor = CrossModalPredictor(
            in_channels=lidar_channels,
            out_channels=camera_channels,
            hidden_dim=hidden_dim
        )
        
        # ConvFuser
        self.fuser = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        
        # Store the auxiliary modules in a list for easy toggling
        self.aux_modules = [
            self.camera_ae, 
            self.lidar_ae, 
            self.camera_predictor, 
            self.lidar_predictor
        ]
    
    def freeze_auxiliary(self):
        for module in self.aux_modules:
            for param in module.parameters():
                param.requires_grad = False
            module.eval() # Must set to eval mode so BatchNorm running stats do not update

    def unfreeze_auxiliary(self):
        for module in self.aux_modules:
            for param in module.parameters():
                param.requires_grad = True
            module.train()
    
    def forward(self, inputs: List[torch.Tensor], return_aux_losses: bool = False):
        camera_features = inputs[0]
        lidar_features = inputs[1]
        
        # Spatial Error Maps (B, 1, H, W)
        cam_self_err = self.camera_ae(camera_features)
        cam_cross_err = self.camera_predictor(camera_features, lidar_features)
        
        lidar_self_err = self.lidar_ae(lidar_features)
        lidar_cross_err = self.lidar_predictor(lidar_features, camera_features)
        
        w_cam = F.sigmoid(self.bias_cam + self.alpha_cam * lidar_self_err - self.beta_cam * cam_cross_err)
        w_lidar = F.sigmoid(self.bias_lidar + self.alpha_lidar * cam_self_err - self.beta_lidar * lidar_cross_err)
        
        weighed_inputs = torch.cat([w_cam * camera_features, w_lidar * lidar_features], dim=1)
        fused_features = self.fuser(weighed_inputs)
        
        if return_aux_losses:
            aux_losses = {
                "loss_cam_self": cam_self_err.mean(),
                "loss_cam_cross": cam_cross_err.mean(),
                "loss_lidar_self": lidar_self_err.mean(),
                "loss_lidar_cross": lidar_cross_err.mean()
            }
            return fused_features, aux_losses
        
        return fused_features
    