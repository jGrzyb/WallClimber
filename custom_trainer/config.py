"""
Encoder-configuration dataclasses and YAML loader.

A configuration file describes how one or more observation specs should be
split into named slices, each handled by a particular encoder type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml


@dataclass
class ObsSplitConfig:
    """How to encode a contiguous slice of a single observation vector."""

    name: str
    start: int
    size: int
    encoder: str  # "linear" | "conv1d" | any key registered in ENCODER_REGISTRY
    normalize: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObsEncoderConfig:
    """Encoding plan for one ``ObservationSpec`` (identified by index)."""

    obs_index: int = 0
    splits: List[ObsSplitConfig] = field(default_factory=list)


@dataclass
class EncoderConfig:
    """
    Top-level configuration that maps observation-spec indices to split /
    encoder descriptions.  Loaded from a small YAML file.

    Supported YAML shapes::

        # Flat (single obs, obs_index defaults to 0)
        obs_index: 0          # optional
        splits:
          - name: proprio
            start: 0
            size: 10
            encoder: linear
          - name: vision
            start: 10
            size: 32
            encoder: conv1d
            params:
              channels: [16, 32]
              kernel_size: 5

        # Multi-obs (list under top-level key "encoders")
        encoders:
          - obs_index: 0
            splits: [...]
          - obs_index: 1
            splits: [...]
    """

    encoders: List[ObsEncoderConfig] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> EncoderConfig:
        with open(path) as f:
            data = yaml.safe_load(f)

        raw_list = data.get("encoders", [data])

        encoders: List[ObsEncoderConfig] = []
        for entry in raw_list:
            splits = [
                ObsSplitConfig(
                    name=s["name"],
                    start=s["start"],
                    size=s["size"],
                    encoder=s["encoder"],
                    normalize=s.get("normalize", True),
                    params=s.get("params", {}),
                )
                for s in entry.get("splits", [])
            ]
            encoders.append(
                ObsEncoderConfig(
                    obs_index=entry.get("obs_index", 0),
                    splits=splits,
                )
            )
        return cls(encoders=encoders)
