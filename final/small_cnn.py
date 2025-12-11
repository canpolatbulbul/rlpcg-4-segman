# small_cnn.py
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SmallGridCNN(BaseFeaturesExtractor):
    """
    Tiny CNN for small grids (e.g., 13x13) with channels-last obs.
    SB3 will convert to channels-first (B, C, H, W) before calling forward().
    """
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        h, w, c = observation_space.shape  # channels-last in env
        self.c = c
        self.h = h
        self.w = w

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            dummy = th.zeros((1, c, h, w))
            n_flat = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU()
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        if obs.dim() == 4 and obs.shape[1] != self.c and obs.shape[-1] == self.c:
            obs = obs.permute(0, 3, 1, 2)  # (B,H,W,C) -> (B,C,H,W)
        x = self.cnn(obs)
        return self.linear(x)
