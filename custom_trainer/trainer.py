"""
Custom PPO trainer and optimizer that wire up the configurable observation
encoders defined in :mod:`custom_trainer.networks`.
"""

from typing import Any, Dict, Optional, Union, cast

import attr

from mlagents.torch_utils import torch, default_device
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.ppo.optimizer_torch import PPOSettings, TorchPPOOptimizer
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.settings import TrainerSettings

from .config import EncoderConfig
from .networks import CustomSimpleActor, CustomSharedActorCritic, CustomValueNetwork

TRAINER_NAME = "custom_ppo"


# ------------------------------------------------------------------ #
# Settings
# ------------------------------------------------------------------ #


@attr.s(auto_attribs=True)
class CustomPPOSettings(PPOSettings):
    """Standard PPO hyper-parameters plus the path to an encoder-config YAML.

    Set ``encoder_config`` to an empty string (the default) to fall back to
    stock ML-Agents behaviour.
    """

    encoder_config: str = ""


# ------------------------------------------------------------------ #
# Optimizer
# ------------------------------------------------------------------ #


class CustomTorchPPOOptimizer(TorchPPOOptimizer):
    """PPO optimizer that uses :class:`CustomValueNetwork` as the critic when
    a custom encoder config is provided."""

    def __init__(
        self,
        policy: TorchPolicy,
        trainer_settings: TrainerSettings,
        encoder_config: Optional[EncoderConfig] = None,
    ):
        super().__init__(policy, trainer_settings)

        if encoder_config is not None and not self.hyperparameters.shared_critic:
            reward_names = [
                key.value for key, _ in trainer_settings.reward_signals.items()
            ]
            self._critic = CustomValueNetwork(
                reward_names,
                policy.behavior_spec.observation_specs,
                network_settings=trainer_settings.network_settings,
                encoder_config=encoder_config,
            )
            self._critic.to(default_device())

            params = list(self.policy.actor.parameters()) + list(
                self._critic.parameters()
            )
            self.optimizer = torch.optim.Adam(
                params, lr=trainer_settings.hyperparameters.learning_rate
            )


# ------------------------------------------------------------------ #
# Trainer
# ------------------------------------------------------------------ #


class CustomPPOTrainer(PPOTrainer):
    """PPO trainer that builds actors / critics with custom observation
    encoders described by a YAML file referenced in the hyper-parameters."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._encoder_config: Optional[EncoderConfig] = None

    @property
    def _loaded_encoder_config(self) -> Optional[EncoderConfig]:
        if self._encoder_config is None:
            hp = cast(CustomPPOSettings, self.trainer_settings.hyperparameters)
            if hp.encoder_config:
                self._encoder_config = EncoderConfig.from_yaml(hp.encoder_config)
        return self._encoder_config

    # -- policy creation ----------------------------------------------- #

    def create_policy(
        self,
        parsed_behavior_id: BehaviorIdentifiers,
        behavior_spec: BehaviorSpec,
    ) -> TorchPolicy:
        enc_cfg = self._loaded_encoder_config

        actor_kwargs: Dict[str, Any] = {
            "conditional_sigma": False,
            "tanh_squash": False,
            "encoder_config": enc_cfg,
        }

        if self.shared_critic:
            actor_cls: type = CustomSharedActorCritic
            actor_kwargs["stream_names"] = [
                key.value
                for key, _ in self.trainer_settings.reward_signals.items()
            ]
        else:
            actor_cls = CustomSimpleActor

        return TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls,
            actor_kwargs,
        )

    # -- optimizer creation -------------------------------------------- #

    def create_optimizer(self) -> TorchOptimizer:
        return CustomTorchPPOOptimizer(
            cast(TorchPolicy, self.policy),
            self.trainer_settings,
            encoder_config=self._loaded_encoder_config,
        )

    # -- name ---------------------------------------------------------- #

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME
