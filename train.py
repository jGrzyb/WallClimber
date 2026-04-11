#!/usr/bin/env python
"""
Train with custom observation encoders.

This is a thin wrapper around ``mlagents-learn`` that registers the
``custom_ppo`` trainer type before the standard ML-Agents CLI starts.

Usage (from the project root)::

    python train.py Assets/climber_custom.yaml --run-id=CustomClimber
    python train.py Assets/climber_custom.yaml --run-id=CustomClimber --force
    python train.py Assets/climber_custom.yaml --run-id=CustomClimber --resume

All arguments after the config file are forwarded to ``mlagents-learn``.
"""

from custom_trainer import register

register()

from mlagents.trainers.learn import main  # noqa: E402

if __name__ == "__main__":
    main()
