"""
Reusable PyTorch encoder modules for observation slices.

New encoder types can be added by:
  1. Implementing an ``nn.Module`` with signature ``(input_size, output_size, **params)``.
  2. Registering it in :data:`ENCODER_REGISTRY`.
"""

from __future__ import annotations

from typing import List, Optional

from mlagents.torch_utils import torch, nn


# ------------------------------------------------------------------ #
# Circular-padding helper
# ------------------------------------------------------------------ #


class CircularPad1d(nn.Module):
    """Wraps the last ``p`` elements to the front and the first ``p`` to
    the back, giving true circular (periodic) boundary conditions for 1-D
    convolutions over angular / cyclical signals."""

    def __init__(self, padding: int):
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [x[..., -self.padding :], x, x[..., : self.padding]], dim=-1
        )


# ------------------------------------------------------------------ #
# Conv1D encoder with circular padding
# ------------------------------------------------------------------ #


class Conv1DCircularEncoder(nn.Module):
    """1-D convolutional encoder with circular (wrap-around) padding.

    Designed for angular / cyclical observation vectors such as radar-style
    distance bins.  Circular padding ensures that the first and last bins
    are treated as neighbours.

    Parameters
    ----------
    input_size : int
        Length of the 1-D input signal (e.g. 32 angular bins).
    output_size : int
        Dimensionality of the output embedding.
    channels : list[int], optional
        Number of output channels for each conv layer (default ``[16, 32]``).
    kernel_size : int, optional
        Convolution kernel width (default ``5``).  Must be odd so that
        circular padding is symmetric.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        channels: Optional[List[int]] = None,
        kernel_size: int = 5,
    ):
        super().__init__()
        if channels is None:
            channels = [16, 32]

        conv_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            pad = kernel_size // 2
            conv_layers.extend(
                [
                    CircularPad1d(pad),
                    nn.Conv1d(in_ch, out_ch, kernel_size),
                    nn.LeakyReLU(),
                ]
            )
            in_ch = out_ch

        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(4)
        flat_size = channels[-1] * 4
        self.fc = nn.Sequential(
            nn.Linear(flat_size, output_size),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_size)
        x = x.unsqueeze(1)  # -> (batch, 1, input_size)
        x = self.conv(x)  # -> (batch, C, input_size)  spatial size preserved
        x = self.pool(x)  # -> (batch, C, 4)
        x = x.reshape(x.size(0), -1)  # -> (batch, C*4)
        return self.fc(x)  # -> (batch, output_size)


# ------------------------------------------------------------------ #
# Registry
# ------------------------------------------------------------------ #

ENCODER_REGISTRY = {
    "conv1d": Conv1DCircularEncoder,
}
