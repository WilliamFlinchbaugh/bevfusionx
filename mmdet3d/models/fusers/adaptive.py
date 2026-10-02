from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["GatedFuser"]


class AdaptiveFuserBase(nn.Module):
    """
    Abstract base class for adaptive weighted fusion
    
    Handles channel projection, modality-dropout, and logging of the gates
    Subclasses implement `_compute_weights`

    The computed weights are stored on ``self.last_weights`` so a training
    script can log them over an epoch for verification that they're doing something
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        modality_dropout: float = 0.0,
        sensor_names: List[str] = None,
    ) -> None:
        super().__init__()
        assert len(in_channels) >= 1
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.modality_dropout = modality_dropout
        self.sensor_names = sensor_names or [
            f"sensor_{i}" for i in range(len(in_channels))
        ]

        self.transforms = nn.ModuleList()
        for c in self.in_channels:
            self.transforms.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )
            )

        # filled in by forward(), handy for logging/visualization
        self.last_weights = []

    def _project(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return [t(x) for t, x in zip(self.transforms, inputs)]

    def _maybe_dropout(self, weights: List[torch.Tensor]) -> List[torch.Tensor]:
        """Randomly zero one modality during training.

        Modality-dropout keeps the gate from collapsing onto a single
        sensor (otherwise it can learn "just always trust lidar").
        """
        # see if we should use modality dropout
        if self.training and self.modality_dropout > 0 and torch.rand(1).item() < self.modality_dropout:
            
            # pick a random modality
            k = torch.randint(0, len(weights), (1,)).item()
            
            # zero out the weights for that modality
            weights = [w if i != k else torch.zeros_like(w) for i, w in enumerate(weights)]
        
        return weights

    def _fuse(
        self, projected: List[torch.Tensor], weights: List[torch.Tensor]
    ) -> torch.Tensor:
        """Weighted sum with per-sample scalar weights (B,) each."""
        stacked = torch.stack(projected, dim=-1)  # B, C, H, W, L
        w = torch.stack(weights, dim=-1)          # B, L
        w = w.view(w.shape[0], 1, 1, 1, -1)      # B, 1, 1, 1, L
        return (stacked * w).sum(dim=-1)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        projected = self._project(inputs)
        weights = self._compute_weights(inputs, projected)
        weights = self._maybe_dropout(weights)
        self.last_weights = [w.detach() for w in weights]
        return self._fuse(projected, weights)

    def _compute_weights(
        self, inputs: List[torch.Tensor], projected: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        raise NotImplementedError


@FUSERS.register_module()
class GatedFuser(AdaptiveFuserBase):
    """Per-sample learned gates over modality features.

    A small MLP sees pooled global context from each sensor's BEV map
    (both mean-pool and max-pool) and outputs one softmax weight per
    sensor. Because the gate input describes the actual feature quality of
    the current scene, the model can dynamically trust camera vs. lidar
    (e.g. camera features are weak in darkness or glare; lidar degrades in
    heavy rain or dust).

    Config:
        type: GatedFuser
        in_channels: [80, 256]   # one entry per modality
        out_channels: 256
        hidden_channels: 64      # MLP width
        dropout: 0.3             # modality-dropout probability
        temperature: 1.0         # <1 sharpens, >1 flattens the softmax
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        hidden_channels: int = 64,
        dropout: float = 0.0,
        temperature: float = 1.0,
        gate_bias_init: float = -2.0,
        sensor_names: List[str] = None,
    ) -> None:
        super().__init__(in_channels, out_channels, dropout, sensor_names)
        self.temperature = temperature
        gate_in = 2 * len(in_channels) * out_channels  # mean+max pooling per sensor
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, len(in_channels)),
        )
        
        # initialize the bias and weights
        nn.init.constant_(self.gate[-1].bias, gate_bias_init)
        with torch.no_grad():
            self.gate[-1].weight.zero_()

    def _compute_weights(self, inputs, projected):
        ctx = []
        for x in projected:
            ctx.append(torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1))
        logits = self.gate(torch.cat(ctx, dim=1)) / self.temperature  # B, L
        weights = torch.softmax(logits, dim=1)  # per-sample, sums to 1
        return list(weights.unbind(dim=1))

