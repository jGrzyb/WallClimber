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

Progress: a tqdm bar tracks environment steps parsed from ML-Agents summary logs.
Disable with env ``WALLCLIMBER_NO_TQDM=1``. Milestone lines print at ~10% intervals of
``max_steps`` from the YAML (when set).

``train_settings`` in the YAML (consumed by this script, not forwarded to mlagents-learn):
  initialize_from: ""   # run-id or path to .onnx/.pt checkpoint; injects --initialize-from
  device: ""            # PyTorch device ("cpu", "cuda", …); injects --torch-device
"""

from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
from pathlib import Path

from custom_trainer import register

register()

import yaml
from tqdm import tqdm

STEP_RE = re.compile(r"Step:\s*(\d+)")
MEAN_REWARD_RE = re.compile(r"Mean Reward:\s*([0-9.eE+-]+)")


# ------------------------------------------------------------------ #
# YAML helpers
# ------------------------------------------------------------------ #

def _resolve_yaml_config_path() -> Path | None:
    """Same trainer file ML-Agents uses: first ``*.yaml`` / ``*.yml`` in argv that exists."""
    for arg in sys.argv[1:]:
        if not arg.endswith((".yaml", ".yml")):
            continue
        p = Path(arg)
        if p.is_file():
            return p
        q = Path.cwd() / arg
        if q.is_file():
            return q
    return None


def _load_yaml(config_path: Path) -> dict:
    try:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[train] WARNING: could not read YAML: {e}", file=sys.stderr)
        return {}


# ------------------------------------------------------------------ #
# train_settings helpers
# ------------------------------------------------------------------ #

def _resolve_initialize_from(value: str) -> tuple[str, str]:
    """
    Return ``(run_id, description)`` for the ``--initialize-from`` CLI flag.

    Accepts:
    - A bare run-id string (no file extension, no path separators):
        "CustomClimber"  →  run_id="CustomClimber"
    - A path to a ``.onnx`` or ``.pt`` file:
        "results/CustomClimber/Climber/Climber-100500.onnx"
        The corresponding ``.pt`` checkpoint is located and the run-id is
        inferred from the path structure ``results/<run-id>/…``.
    """
    p = Path(value)

    if p.suffix not in (".onnx", ".pt") and not any(c in value for c in ("/", "\\")):
        # Bare run-id — use directly.
        return value, f"run-id '{value}'"

    # Resolve relative paths from cwd.
    if not p.is_absolute():
        p = Path.cwd() / p

    if not p.exists():
        raise FileNotFoundError(
            f"[train] initialize_from: file not found: {p}\n"
            f"  Make sure the path is correct (relative to the project root)."
        )

    # For .onnx: find the companion .pt checkpoint.
    if p.suffix == ".onnx":
        # Try <same stem>.pt in the same directory.
        pt = p.with_suffix(".pt")
        if not pt.exists():
            # Try checkpoint.pt in the same directory (most-recent checkpoint).
            pt = p.parent / "checkpoint.pt"
        if not pt.exists():
            raise FileNotFoundError(
                f"[train] initialize_from: could not find a .pt checkpoint alongside {p}\n"
                f"  Expected {p.with_suffix('.pt')} or {p.parent / 'checkpoint.pt'}"
            )
    else:
        pt = p

    # Infer run-id from path: look for a "results" ancestor.
    parts = pt.parts
    run_id = None
    for i, part in enumerate(parts):
        if part.lower() == "results" and i + 1 < len(parts):
            run_id = parts[i + 1]
            break

    if run_id is None:
        raise ValueError(
            f"[train] initialize_from: could not infer run-id from path: {pt}\n"
            f"  Path must be under a 'results/<run-id>/' directory,\n"
            f"  or specify the run-id directly (e.g. initialize_from: CustomClimber)."
        )

    return run_id, f"'{value}' → run-id '{run_id}' (checkpoint: {pt.name})"


def _apply_train_settings(cfg: dict) -> None:
    """
    Read ``train_settings`` from the loaded YAML and inject any required
    flags into ``sys.argv`` before ``mlagents-learn`` parses the command line.
    """
    ts = cfg.get("train_settings") or {}

    # --initialize-from
    init_from = str(ts.get("initialize_from") or "").strip()
    if init_from:
        # Don't add if the user already passed --initialize-from on the CLI.
        if not any(a.startswith("--initialize-from") for a in sys.argv):
            try:
                run_id, desc = _resolve_initialize_from(init_from)
                sys.argv.append(f"--initialize-from={run_id}")
                print(f"[train] Warm-starting from {desc}", file=sys.stderr)
            except (FileNotFoundError, ValueError) as exc:
                print(f"[train] WARNING: {exc}", file=sys.stderr)
        else:
            print(
                "[train] --initialize-from already in argv; train_settings.initialize_from ignored.",
                file=sys.stderr,
            )

    # --torch-device
    device = str(ts.get("device") or "").strip()
    if device:
        if not any(a.startswith("--torch-device") for a in sys.argv):
            sys.argv.append(f"--torch-device={device}")
            print(f"[train] PyTorch device (from train_settings.device): {device}", file=sys.stderr)
        else:
            print(
                "[train] --torch-device already in argv; train_settings.device ignored.",
                file=sys.stderr,
            )
    else:
        # Auto-detect and print what ML-Agents will choose.
        try:
            from mlagents.torch_utils import torch  # noqa: PLC0415
            auto = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[train] PyTorch device (auto-detect): {auto}", file=sys.stderr)
        except ImportError:
            pass


# ------------------------------------------------------------------ #
# Config summary
# ------------------------------------------------------------------ #

def _print_training_config_summary(config_path: Path | None) -> None:
    """Echo key fields so edits to the YAML are visible before mlagents-learn starts."""
    if config_path is None:
        print(
            "[train] No trainer .yaml in argv (e.g. python train.py Assets/climber_custom.yaml …). "
            "Milestones and tqdm total may be wrong.",
            file=sys.stderr,
        )
        return
    abs_cfg = config_path.resolve()
    print(f"[train] Trainer YAML (absolute): {abs_cfg}", file=sys.stderr)
    try:
        cfg = _load_yaml(config_path)
    except Exception:
        return
    env = cfg.get("env_settings") or {}
    print(
        f"[train] env_settings: num_envs={env.get('num_envs')!r}, "
        f"num_areas={env.get('num_areas')!r}, env_path={env.get('env_path')!r}",
        file=sys.stderr,
    )
    env_params = cfg.get("environment_parameters")
    if env_params:
        print(
            f"[train] environment_parameters (Unity side channel): {env_params!r}",
            file=sys.stderr,
        )
    behaviors = cfg.get("behaviors") or {}
    for name, spec in behaviors.items():
        if not isinstance(spec, dict):
            continue
        hp = spec.get("hyperparameters") or {}
        enc = hp.get("encoder_config")
        print(
            f"[train] behavior {name!r}: max_steps={spec.get('max_steps')!r}, "
            f"learning_rate={hp.get('learning_rate')!r}, encoder_config={enc!r}",
            file=sys.stderr,
        )
        if enc:
            enc_p = Path(enc)
            if not enc_p.is_file():
                alt = Path.cwd() / enc
                if alt.is_file():
                    enc_p = alt
                else:
                    print(
                        f"[train] WARNING: encoder_config not found from cwd={Path.cwd()!s}: {enc!r}",
                        file=sys.stderr,
                    )
    print(
        "[train] Edit the file above for training; results/*/configuration.yaml is only a run snapshot.",
        file=sys.stderr,
    )


# ------------------------------------------------------------------ #
# TrainingConfig (for tqdm)
# ------------------------------------------------------------------ #

class TrainingConfig:
    """Key values parsed from the trainer YAML for progress tracking."""

    __slots__ = ("max_steps", "summary_freq", "buffer_size")

    def __init__(self, max_steps: int = 0, summary_freq: int = 0, buffer_size: int = 0):
        self.max_steps = max_steps
        self.summary_freq = summary_freq
        self.buffer_size = buffer_size

    @property
    def num_summaries(self) -> int:
        """Expected number of ML-Agents summary log lines (= bar ticks)."""
        if self.max_steps > 0 and self.summary_freq > 0:
            return max(1, self.max_steps // self.summary_freq)
        return 0

    @property
    def num_buffer_fills(self) -> int:
        """Total PPO updates (buffer flushes) across the run."""
        if self.max_steps > 0 and self.buffer_size > 0:
            import math
            return math.ceil(self.max_steps / self.buffer_size)
        return 0


def _load_training_config(config_path: Path | None) -> TrainingConfig:
    if config_path is None:
        return TrainingConfig()
    try:
        cfg = _load_yaml(config_path)
        behaviors = cfg.get("behaviors") or {}
        max_steps = summary_freq = buffer_size = 0
        for spec in behaviors.values():
            if not isinstance(spec, dict):
                continue
            if "max_steps" in spec:
                max_steps = max(max_steps, int(spec["max_steps"]))
            if "summary_freq" in spec:
                summary_freq = max(summary_freq, int(spec["summary_freq"]))
            hp = spec.get("hyperparameters") or {}
            if "buffer_size" in hp:
                buffer_size = max(buffer_size, int(hp["buffer_size"]))
        return TrainingConfig(max_steps=max_steps, summary_freq=summary_freq, buffer_size=buffer_size)
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return TrainingConfig()


# ------------------------------------------------------------------ #
# tqdm / logging
# ------------------------------------------------------------------ #

class TqdmStreamHandler(logging.Handler):
    """Route any log record through tqdm.write so it doesn't overwrite the bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:
            self.handleError(record)


class MlAgentsStepProgressHandler(logging.Handler):
    """
    Drive tqdm from ML-Agents ConsoleWriter summary lines (``Step: N``).

    The bar advances by ONE tick per summary line (every summary_freq env-steps).
    The postfix shows the raw step count, current buffer fill, and mean reward so
    the bar moves at a human-readable pace instead of jumping by tens-of-thousands.
    """

    def __init__(self, pbar: tqdm, cfg: TrainingConfig):
        super().__init__(level=logging.INFO)
        self.pbar = pbar
        self.cfg = cfg
        self._next_milestone_pct = 10

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            m = STEP_RE.search(msg)
            if not m:
                return
            step = int(m.group(1))

            # One tick per summary log, regardless of step delta
            self.pbar.update(1)

            postfix: dict[str, str] = {
                "env step": f"{step:,}",
            }
            if self.cfg.max_steps > 0:
                pct = 100.0 * step / self.cfg.max_steps
                postfix["env step"] = f"{step:,} ({pct:.1f}%)"
            if self.cfg.num_buffer_fills > 0 and self.cfg.buffer_size > 0:
                import math
                fill = math.ceil(step / self.cfg.buffer_size)
                postfix["buf"] = f"{fill}/{self.cfg.num_buffer_fills}"
            rm = MEAN_REWARD_RE.search(msg)
            if rm:
                postfix["reward"] = f"{float(rm.group(1)):.3f}"
            self.pbar.set_postfix(postfix, refresh=True)

            if self.cfg.max_steps > 0:
                while self._next_milestone_pct <= 100:
                    thr = (self._next_milestone_pct / 100.0) * self.cfg.max_steps
                    if step < thr:
                        break
                    tqdm.write(
                        f"[train] ~{self._next_milestone_pct}%  "
                        f"step {step:,} / {self.cfg.max_steps:,}"
                        + (f"  reward {float(rm.group(1)):.3f}" if rm else "")
                    )
                    self._next_milestone_pct += 10
        except Exception:
            self.handleError(record)


def _redirect_mlagents_loggers_through_tqdm() -> None:
    """
    ML-Agents' get_logger() adds a StreamHandler(sys.stdout) to every logger it
    creates.  Those raw writes scroll past tqdm's bar and overwrite it.  We swap
    every such handler for a TqdmStreamHandler so all log output is routed through
    tqdm.write(), which knows how to preserve the bar.
    """
    import mlagents_envs.logging_util as lu  # noqa: PLC0415

    tqdm_handler = TqdmStreamHandler(level=logging.DEBUG)
    tqdm_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # Replace handlers on loggers already created by get_logger()
    for logger in list(lu._loggers):
        _swap_stdout_handlers(logger, tqdm_handler)

    # Monkey-patch get_logger so future loggers also use tqdm.write
    _orig_get_logger = lu.get_logger

    def _patched_get_logger(name: str) -> logging.Logger:
        logger = _orig_get_logger(name)
        _swap_stdout_handlers(logger, tqdm_handler)
        return logger

    lu.get_logger = _patched_get_logger


def _swap_stdout_handlers(logger: logging.Logger, replacement: logging.Handler) -> None:
    for h in logger.handlers[:]:
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout:
            logger.removeHandler(h)
            if replacement not in logger.handlers:
                logger.addHandler(replacement)


def _install_progress(config_path: Path | None) -> tqdm | None:
    if os.environ.get("WALLCLIMBER_NO_TQDM", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return None

    cfg = _load_training_config(config_path)
    # Bar total = number of summary log lines expected (one tick per log).
    # Falls back to None (unknown total) if summary_freq is missing.
    total = cfg.num_summaries or None

    desc = "Training"
    if cfg.num_summaries and cfg.summary_freq:
        desc = f"Training  (1 tick = {cfg.summary_freq:,} env-steps)"

    pbar: tqdm = tqdm(
        total=total,
        desc=desc,
        unit="summary",
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=0.5,
        position=0,
        leave=True,
    )
    pbar.refresh()

    step_handler = MlAgentsStepProgressHandler(pbar, cfg=cfg)
    # Logger may not exist yet; Python's registry preserves handlers across imports.
    logging.getLogger("mlagents.trainers.stats").addHandler(step_handler)

    return pbar


# ------------------------------------------------------------------ #
# YAML scrubbing — remove keys unknown to mlagents-learn
# ------------------------------------------------------------------ #

# Top-level keys consumed exclusively by train.py.
# They must be stripped before ML-Agents parses the config.
_TRAIN_ONLY_KEYS = {"train_settings"}


def _write_scrubbed_yaml(config_path: Path, cfg: dict) -> Path:
    """
    Write a copy of ``cfg`` with :data:`_TRAIN_ONLY_KEYS` removed to a
    temporary file and return its path.  The caller is responsible for
    deleting the file when done.
    """
    cleaned = {k: v for k, v in cfg.items() if k not in _TRAIN_ONLY_KEYS}
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="wallclimber_",
        delete=False,
        encoding="utf-8",
    )
    yaml.dump(cleaned, tmp, allow_unicode=True, default_flow_style=False)
    tmp.close()
    return Path(tmp.name)


def _swap_yaml_in_argv(original: Path, replacement: Path) -> None:
    """Replace the first occurrence of ``original`` in ``sys.argv`` with ``replacement``."""
    orig_str = str(original)
    orig_abs = str(original.resolve())
    for i, arg in enumerate(sys.argv):
        if arg == orig_str or arg == orig_abs or Path(arg).resolve() == original.resolve():
            sys.argv[i] = str(replacement)
            return


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main() -> None:
    cfg_path = _resolve_yaml_config_path()
    _print_training_config_summary(cfg_path)

    cfg_data: dict = {}
    tmp_yaml: Path | None = None

    if cfg_path is not None:
        cfg_data = _load_yaml(cfg_path)

        # Apply train_settings (injects --initialize-from / --torch-device into sys.argv).
        _apply_train_settings(cfg_data)

        # Scrub train-only keys so ML-Agents doesn't reject them.
        if any(k in cfg_data for k in _TRAIN_ONLY_KEYS):
            tmp_yaml = _write_scrubbed_yaml(cfg_path, cfg_data)
            _swap_yaml_in_argv(cfg_path, tmp_yaml)

    pbar = _install_progress(cfg_path)
    try:
        from mlagents.trainers.learn import main as mlearn_main  # noqa: PLC0415

        # All mlagents loggers are now created; redirect their stdout handlers
        # through tqdm.write so they don't overwrite the progress bar.
        if pbar is not None:
            _redirect_mlagents_loggers_through_tqdm()

        mlearn_main()
    finally:
        if pbar is not None:
            pbar.close()
        if tmp_yaml is not None:
            try:
                tmp_yaml.unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    main()
