# custom_trainer — Custom Observation Encoders for ML-Agents

A drop-in extension module for Unity ML-Agents that lets you define **per-slice
observation encoders** without modifying any installed packages.  Observations
from a single `VectorSensor` can be split by index range and routed through
different PyTorch encoder architectures (linear, Conv1D, or your own), while
the rest of the PPO training pipeline — checkpointing, ONNX export, curriculum,
reward signals — works exactly as before.

## Architecture overview

```
┌──────────────────────────────────────────────────────────┐
│  Vector observation  (e.g. 42 floats)                    │
│  ┌────────────┐ ┌────────────────────────────────────┐   │
│  │ 0..9       │ │ 10..41                             │   │
│  │ proprio    │ │ vision bins (32 angular distances)  │   │
│  └─────┬──────┘ └──────────────┬─────────────────────┘   │
│        │                       │                         │
│  VectorInput             Conv1DCircular                  │
│  (normalize)             Encoder                         │
│        │                       │                         │
│        │  (10)                 │  (h_size)               │
│        └──────────┬────────────┘                         │
│              concat (10 + 256)                           │
│                   │                                      │
│            LinearEncoder (body MLP)                      │
│                   │                                      │
│              ActionModel / ValueHeads                    │
└──────────────────────────────────────────────────────────┘
```

Both the **actor** and the **critic** use the same split-encoder architecture
(initialised independently), so the convolutional features are trained
end-to-end through PPO's policy and value losses.

## File layout

```
custom_trainer/
├── __init__.py      Plugin entry-point & manual register() helper
├── config.py        Dataclasses + YAML loader for encoder configs
├── encoders.py      Conv1DCircularEncoder, CircularPad1d, encoder registry
├── networks.py      SplitObservationEncoder, CustomNetworkBody, actor/critic
├── trainer.py       CustomPPOTrainer, CustomPPOSettings, CustomTorchPPOOptimizer
└── README.md        This file

Assets/
├── climber_encoder.yaml   Encoder architecture for the Climber agent
├── climber_custom.yaml    Training YAML that selects custom_ppo

train.py                   Convenience entry-point (registers plugin, runs CLI)
```

## Quick start

### 1. Activate the virtual environment

```bash
# If you haven't yet installed mlagents:
uv pip install -e ".[mlagents]"
```

### 2. Launch training

```bash
python train.py Assets/climber_custom.yaml --run-id=CustomClimber --force
```

Press Play in the Unity Editor (the scene with the Climber agent) after the
script prints `Listening on port 5005`.  Training logs go to `results/`.

### 3. Resume / change runs

```bash
python train.py Assets/climber_custom.yaml --run-id=CustomClimber --resume
```

## How it works

### Registration

`custom_trainer` plugs into ML-Agents via two mechanisms (use either one):

| Method | How | When |
|--------|-----|------|
| **`train.py`** (recommended) | Calls `custom_trainer.register()` then `mlagents.trainers.learn.main()` | Always works, no install step |
| **entry-points** | `pyproject.toml` declares `[project.entry-points."mlagents.trainer_type"]` | Works after `uv pip install -e .` (or `pip install -e .`) |

Both populate `mlagents.plugins.all_trainer_types` with `"custom_ppo"` →
`CustomPPOTrainer` before the CLI parses the YAML config.

### Training YAML

Use `trainer_type: custom_ppo` and add an `encoder_config` field under
`hyperparameters` that points to the encoder YAML:

```yaml
behaviors:
  Climber:
    trainer_type: custom_ppo
    hyperparameters:
      batch_size: 2048
      buffer_size: 20480
      learning_rate: 0.0003
      beta: 0.005
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
      learning_rate_schedule: linear
      encoder_config: Assets/climber_encoder.yaml   # <-- path to encoder spec
    network_settings:
      normalize: true
      hidden_units: 256
      num_layers: 3
    reward_signals:
      extrinsic:
        gamma: 0.995
        strength: 1.0
    max_steps: 10000000
    time_horizon: 1000
    summary_freq: 30000
```

All other fields (`reward_signals`, `network_settings`, `max_steps`, etc.)
work identically to standard `ppo`.

### Encoder config YAML

The encoder config tells the trainer how to split a vector observation and
which encoder to apply to each part:

```yaml
obs_index: 0          # which ObservationSpec to split (0 = first)

splits:
  - name: proprioception
    start: 0
    size: 10
    encoder: linear     # pass-through with optional normalisation
    normalize: true

  - name: vision
    start: 10
    size: 32
    encoder: conv1d     # circular 1-D convolution
    normalize: false
    params:
      channels: [16, 32]
      kernel_size: 5
```

**Fields per split:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Human-readable label (for logging / debugging) |
| `start` | int | Start index in the observation vector |
| `size` | int | Number of floats in this slice |
| `encoder` | str | `"linear"` or `"conv1d"` (or any key in `ENCODER_REGISTRY`) |
| `normalize` | bool | Whether to apply running-mean normalisation (only for `linear`) |
| `params` | dict | Extra keyword arguments forwarded to the encoder constructor |

### Multiple observation specs

If the agent exposes several sensors (multiple `ObservationSpec`s), wrap the
entries in a top-level `encoders` list:

```yaml
encoders:
  - obs_index: 0
    splits:
      - { name: limbs, start: 0, size: 10, encoder: linear }
      - { name: radar, start: 10, size: 32, encoder: conv1d, params: { channels: [16,32], kernel_size: 5 } }
  - obs_index: 1
    splits:
      - { name: camera, start: 0, size: 1024, encoder: conv1d, params: { channels: [32,64], kernel_size: 7 } }
```

Observation specs **not** listed in the config are encoded with the default
`VectorInput` (pass-through + optional normalisation).

## Adding a new encoder type

1. Create an `nn.Module` whose `__init__` takes `(input_size, output_size, **params)`:

```python
# custom_trainer/encoders.py
class MyFancyEncoder(nn.Module):
    def __init__(self, input_size: int, output_size: int, depth: int = 3):
        super().__init__()
        # ... build layers ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_size)
        # return shape: (batch, output_size)
        ...
```

2. Register it:

```python
# custom_trainer/encoders.py
ENCODER_REGISTRY["fancy"] = MyFancyEncoder
```

3. Reference it in the encoder YAML:

```yaml
splits:
  - name: my_obs
    start: 0
    size: 64
    encoder: fancy
    params:
      depth: 5
```

## Conv1D circular encoder details

`Conv1DCircularEncoder` is purpose-built for angular / cyclical observation
vectors (e.g. radar-style distance bins that wrap around 360°).

- **Circular padding** ensures the first and last bins are treated as
  neighbours, eliminating boundary artefacts that regular zero-padding would
  introduce.
- The default architecture (for 32 bins) is:

```
(batch, 32) → unsqueeze → (batch, 1, 32)
  CircularPad(2) → Conv1d(1→16, k=5) → LeakyReLU
  CircularPad(2) → Conv1d(16→32, k=5) → LeakyReLU
  AdaptiveAvgPool1d(4) → (batch, 32, 4)
  Flatten → (batch, 128)
  Linear(128 → h_size) → LeakyReLU → (batch, h_size)
```

- Total parameters: ~35 K (much lighter than a full image CNN).
- The spatial pooling to 4 bins preserves coarse directional information while
  keeping the parameter count low.

## Parallel training (WallClimber)

The WallClimber project trains **four** agents in parallel by default:

- **Python** (`Assets/climber.yaml`, `Assets/climber_custom.yaml`): `env_settings`
  with `num_areas: 4` and `num_envs: 1`.  When `env_path` is `null` (training
  from the Unity Editor), ML-Agents **requires** `num_envs: 1`; multi-process
  parallelism uses `num_envs` with a **built executable** instead.
- **Unity layout** (`Assets/Scripts/WallClimberParallelTrainingBootstrap.cs` +
  `WallClimberHorizontalAreaReplicator.cs`): groups `Ground`, `GripGrid`,
  `Climber`, and `Square` under a `TrainingArea` root and duplicates that root
  along **+X** (side-by-side), not a 2×2 grid.  Keep `num_areas` in YAML aligned
  with `WallClimberParallelTrainingBootstrap.DefaultNumAreas`.
- **Camera** (`Assets/Scripts/MyCamera.cs`): the training scene
  `Assets/Scenes/WallClimberGrips.unity` overrides the Main Camera prefab to
  `WallClimberCameraMode.StaticTrainingOverview` so the orthographic view stays
  fixed and frames every climber.  Adjust `framingPadding` on the camera if the
  edges are tight.
- **Inference** (`Assets/Scenes/WallClimberInference.unity`): separate scene with
  an `InferenceSceneRoot` carrying `WallClimberInferenceScene` so parallel
  bootstrap is **not** auto-created; the camera uses **FollowAgent** (prefab
  default).  Open this scene for play-mode inference / heuristics with a single
  agent and follow camera.

If you change the number of areas, update **both** the YAML and the C# default
(or a scene-placed instance of the bootstrap component).

## Compatibility

- **Python**: 3.11+ (matches the project's `.python-version`)
- **ML-Agents**: 1.1.0 (the version pinned in `pyproject.toml`)
- **No package modifications**: the module only subclasses public ML-Agents
  classes and uses the documented plugin system.
- **ONNX export**: `CustomSimpleActor.forward()` inherits `SimpleActor`'s
  export contract, so `ModelSerializer` works unchanged.
- **Checkpointing / resume**: state dicts include the custom encoder weights
  automatically (they are `nn.Module` sub-modules of the actor and critic).

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `KeyError: 'custom_ppo'` | Make sure you run via `python train.py ...` (not `mlagents-learn` directly), **or** reinstall with `uv pip install -e ".[mlagents]"` so entry-points are registered. |
| `FileNotFoundError` on encoder config | The `encoder_config` path is resolved relative to the **working directory**. Run from the project root. |
| Shape mismatch at runtime | Double-check that the `start` + `size` values in the encoder YAML match the observation layout in `Climber.CollectObservations`. |
| `shared_critic: true` with custom encoders | Supported — `CustomSharedActorCritic` replaces the shared network body the same way. |
