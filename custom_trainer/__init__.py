"""
Custom ML-Agents trainer with configurable observation encoders.

Supports splitting vector observations and routing different slices through
different encoder architectures (e.g. Conv1D for angular vision, linear for
proprioception), while remaining fully compatible with the standard PPO
training loop, checkpointing, and ONNX export.

Quick start:
    from custom_trainer import register
    register()
    from mlagents.trainers.learn import main
    main()
"""

from .trainer import CustomPPOTrainer, CustomPPOSettings


def get_custom_trainer_types():
    """
    Plugin entry point for ``mlagents.trainer_type``.

    Returns ``(trainer_types_dict, trainer_settings_dict)`` as expected by
    :func:`mlagents.plugins.trainer_type.register_trainer_plugins`.
    """
    import cattr

    cattr.register_structure_hook(str, lambda v, _: str(v))

    trainer_types = {CustomPPOTrainer.get_trainer_name(): CustomPPOTrainer}
    trainer_settings = {CustomPPOTrainer.get_trainer_name(): CustomPPOSettings}
    return trainer_types, trainer_settings


def register():
    """
    Manually inject the custom trainer into ML-Agents' plugin registry.

    Call this **before** ``mlagents.trainers.learn.main()`` when you are not
    relying on ``pyproject.toml`` entry-points for discovery.
    """
    import cattr
    from mlagents import plugins as mla_plugins

    # The version of cattr bundled with ML-Agents does not ship a structure
    # hook for plain ``str``.  Register one so that our ``encoder_config: str``
    # field survives YAML → attrs deserialization.
    cattr.register_structure_hook(str, lambda v, _: str(v))

    trainer_types, trainer_settings = get_custom_trainer_types()
    mla_plugins.all_trainer_types.update(trainer_types)
    mla_plugins.all_trainer_settings.update(trainer_settings)
