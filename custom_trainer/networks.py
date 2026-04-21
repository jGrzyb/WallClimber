"""
Custom network components that replace ML-Agents' stock observation encoder
with a configurable split-and-encode pipeline.

Hierarchy
---------
SplitObservationEncoder   replaces   ObservationEncoder
CustomNetworkBody         replaces   NetworkBody
CustomSimpleActor         extends    SimpleActor
CustomSharedActorCritic   extends    SharedActorCritic
CustomValueNetwork        replaces   ValueNetwork
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from mlagents.torch_utils import torch, nn
from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.encoders import VectorInput
from mlagents.trainers.torch_entities.layers import LinearEncoder, LSTM
from mlagents.trainers.torch_entities.decoders import ValueHeads
from mlagents.trainers.torch_entities.networks import (
    SimpleActor,
    SharedActorCritic,
    NetworkBody,
    Critic,
)
from mlagents.trainers.trajectory import ObsUtil

from .config import EncoderConfig, ObsSplitConfig
from .encoders import ENCODER_REGISTRY


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _build_split_encoder(
    split: ObsSplitConfig,
    h_size: int,
    default_normalize: bool,
) -> nn.Module:
    """Instantiate the right encoder for one observation split."""
    normalize = split.normalize if split.normalize is not None else default_normalize

    if split.encoder == "linear":
        return VectorInput(split.size, normalize)

    if split.encoder in ENCODER_REGISTRY:
        cls = ENCODER_REGISTRY[split.encoder]
        return cls(input_size=split.size, output_size=h_size, **split.params)

    raise ValueError(
        f"Unknown encoder type '{split.encoder}'. "
        f"Available: 'linear', {list(ENCODER_REGISTRY)}"
    )


def _split_output_size(split: ObsSplitConfig, h_size: int) -> int:
    """VectorInput preserves dimensionality; all others project to h_size."""
    return split.size if split.encoder == "linear" else h_size


# ====================================================================== #
# Observation encoder
# ====================================================================== #


class SplitObservationEncoder(nn.Module):
    """Drop-in replacement for ``ObservationEncoder`` that splits vector
    observations according to an :class:`EncoderConfig` and routes each
    slice through its own encoder module."""

    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        h_size: int,
        encoder_config: EncoderConfig,
        normalize: bool = False,
    ):
        super().__init__()
        self.normalize = normalize
        self._num_obs = len(observation_specs)

        configured_indices = {e.obs_index for e in encoder_config.encoders}

        # --- custom-split encoders ---
        self._split_encoders: nn.ModuleDict = nn.ModuleDict()
        self._split_configs: Dict[int, List[ObsSplitConfig]] = {}

        for enc_cfg in encoder_config.encoders:
            idx = enc_cfg.obs_index
            encoders = nn.ModuleList(
                [_build_split_encoder(s, h_size, normalize) for s in enc_cfg.splits]
            )
            self._split_encoders[str(idx)] = encoders
            self._split_configs[idx] = enc_cfg.splits

        # --- passthrough (default VectorInput) for unconfigured obs ---
        self._pt_indices: List[int] = []
        self._pt_encoders = nn.ModuleList()
        for i, spec in enumerate(observation_specs):
            if i not in configured_indices:
                self._pt_encoders.append(VectorInput(spec.shape[0], normalize))
                self._pt_indices.append(i)

        # --- encoding size ---
        total = 0
        for idx, splits in self._split_configs.items():
            for s in splits:
                total += _split_output_size(s, h_size)
        for pos, obs_idx in enumerate(self._pt_indices):
            total += observation_specs[obs_idx].shape[0]

        self._total_enc_size = total
        self._total_goal_enc_size = 0

    # -- properties ---------------------------------------------------- #

    @property
    def total_enc_size(self) -> int:
        return self._total_enc_size

    @property
    def total_goal_enc_size(self) -> int:
        return self._total_goal_enc_size

    # -- normalization ------------------------------------------------- #

    def update_normalization(self, buffer: AgentBuffer) -> None:
        obs = ObsUtil.from_buffer(buffer, self._num_obs)

        for idx_s, encoders in self._split_encoders.items():
            idx = int(idx_s)
            splits = self._split_configs[idx]
            full = torch.as_tensor(obs[idx].to_ndarray())
            for split, enc in zip(splits, encoders):
                if isinstance(enc, VectorInput):
                    enc.update_normalization(full[:, split.start : split.start + split.size])

        for pos, obs_idx in enumerate(self._pt_indices):
            enc = self._pt_encoders[pos]
            if isinstance(enc, VectorInput):
                enc.update_normalization(torch.as_tensor(obs[obs_idx].to_ndarray()))

    def copy_normalization(self, other: SplitObservationEncoder) -> None:
        if not self.normalize:
            return
        for key in self._split_encoders:
            for a, b in zip(self._split_encoders[key], other._split_encoders[key]):
                if isinstance(a, VectorInput) and isinstance(b, VectorInput):
                    a.copy_normalization(b)
        for a, b in zip(self._pt_encoders, other._pt_encoders):
            if isinstance(a, VectorInput) and isinstance(b, VectorInput):
                a.copy_normalization(b)

    # -- forward ------------------------------------------------------- #

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        parts: List[torch.Tensor] = []

        for idx_s, encoders in self._split_encoders.items():
            idx = int(idx_s)
            obs = inputs[idx]
            for split, enc in zip(self._split_configs[idx], encoders):
                parts.append(enc(obs[:, split.start : split.start + split.size]))

        for pos, obs_idx in enumerate(self._pt_indices):
            parts.append(self._pt_encoders[pos](inputs[obs_idx]))

        return torch.cat(parts, dim=1)

    def get_goal_encoding(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError(
            "Goal conditioning is not supported with custom split encoders."
        )


# ====================================================================== #
# Network body
# ====================================================================== #


class CustomNetworkBody(nn.Module):
    """API-compatible replacement for ``NetworkBody`` that uses
    :class:`SplitObservationEncoder`."""

    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        encoder_config: EncoderConfig,
        encoded_act_size: int = 0,
    ):
        super().__init__()
        self.normalize = network_settings.normalize
        self.use_lstm = network_settings.memory is not None
        self.h_size = network_settings.hidden_units
        self.m_size = (
            network_settings.memory.memory_size
            if network_settings.memory is not None
            else 0
        )

        self.observation_encoder = SplitObservationEncoder(
            observation_specs,
            self.h_size,
            encoder_config,
            self.normalize,
        )

        total_enc = self.observation_encoder.total_enc_size + encoded_act_size
        self._body_endoder = LinearEncoder(
            total_enc, network_settings.num_layers, self.h_size
        )

        if self.use_lstm:
            self.lstm = LSTM(self.h_size, self.m_size)
        else:
            self.lstm = None  # type: ignore

    @property
    def memory_size(self) -> int:
        return self.lstm.memory_size if self.use_lstm else 0

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.observation_encoder.update_normalization(buffer)

    def copy_normalization(self, other: CustomNetworkBody) -> None:
        self.observation_encoder.copy_normalization(other.observation_encoder)

    def forward(
        self,
        inputs: List[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.observation_encoder(inputs)
        if actions is not None:
            encoded = torch.cat([encoded, actions], dim=1)
        encoding = self._body_endoder(encoded)

        if self.use_lstm:
            encoding = encoding.reshape([-1, sequence_length, self.h_size])
            encoding, memories = self.lstm(encoding, memories)
            encoding = encoding.reshape([-1, self.m_size // 2])
        return encoding, memories


# ====================================================================== #
# Actor wrappers
# ====================================================================== #


class CustomSimpleActor(SimpleActor):
    """``SimpleActor`` whose ``network_body`` is replaced by a
    :class:`CustomNetworkBody` when *encoder_config* is provided."""

    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        encoder_config: Optional[EncoderConfig] = None,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        super().__init__(
            observation_specs,
            network_settings,
            action_spec,
            conditional_sigma=conditional_sigma,
            tanh_squash=tanh_squash,
        )
        if encoder_config is not None:
            self.network_body = CustomNetworkBody(
                observation_specs, network_settings, encoder_config
            )


class CustomSharedActorCritic(SharedActorCritic):
    """``SharedActorCritic`` whose ``network_body`` is replaced by a
    :class:`CustomNetworkBody` when *encoder_config* is provided."""

    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        stream_names: List[str],
        encoder_config: Optional[EncoderConfig] = None,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        super().__init__(
            observation_specs,
            network_settings,
            action_spec,
            stream_names,
            conditional_sigma=conditional_sigma,
            tanh_squash=tanh_squash,
        )
        if encoder_config is not None:
            self.network_body = CustomNetworkBody(
                observation_specs, network_settings, encoder_config
            )


# ====================================================================== #
# Critic wrapper
# ====================================================================== #


class CustomValueNetwork(nn.Module, Critic):
    """``ValueNetwork`` whose body is optionally a
    :class:`CustomNetworkBody`."""

    def __init__(
        self,
        stream_names: List[str],
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        encoder_config: Optional[EncoderConfig] = None,
        encoded_act_size: int = 0,
        outputs_per_stream: int = 1,
    ):
        nn.Module.__init__(self)

        if encoder_config is not None:
            self.network_body = CustomNetworkBody(
                observation_specs,
                network_settings,
                encoder_config,
                encoded_act_size=encoded_act_size,
            )
        else:
            self.network_body = NetworkBody(
                observation_specs,
                network_settings,
                encoded_act_size=encoded_act_size,
            )

        encoding_size = (
            network_settings.memory.memory_size // 2
            if network_settings.memory is not None
            else network_settings.hidden_units
        )
        self.value_heads = ValueHeads(stream_names, encoding_size, outputs_per_stream)

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size

    def critic_pass(
        self,
        inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return self.forward(inputs, memories=memories, sequence_length=sequence_length)

    def forward(
        self,
        inputs: List[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        encoding, memories_out = self.network_body(
            inputs, actions, memories, sequence_length
        )
        return self.value_heads(encoding), memories_out
