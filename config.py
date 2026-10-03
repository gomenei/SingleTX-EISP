"""Unified experiment settings. Dependencies: Python >=3.10, numpy, torch, PyYAML.

Run from this folder (data paths are external; no data are bundled)::

    python train.py --config config/mnist_noise30_N16.yaml
    python test.py --config config/mnist_noise30_N16.yaml
    python train.py --dataset mnist --train-data /data/train.pkl --test-data /data/test.pkl --output /runs/mnist
    python train.py --dataset cylinder --train-data /data/train.pkl --test-data /data/test.pkl --output /runs/cylinder
    python train.py --dataset if --train-data /data/synthetic.pkl --test-data /data/FDE.pkl --output /runs/if
    python train.py --dataset 3d --train-data /data/train_npy --test-data /data/test_npy --output /runs/3d
    python test.py --checkpoint /runs/mnist/best.pt --test-data /data/test.pkl --output /results/mnist

MNIST/CYLINDER/IF: trusted pickle with {'params': ..., 'data': ...}.
3D: directory containing E_s_real.npy, E_s_imag.npy, epsilon_gt.npy,
x_dom.npy, y_dom.npy, z_dom.npy; physics/J arrays are needed only for J mode.
IF normally trains on synthetic data and evaluates on measured IF data.
Measured IF evaluation does not require J labels. Incidence-specific IF
receiver operators with shape (incidences,receivers,pixels) are supported.

Default: coordinate MLP directly predicts epsilon. For current reconstruction:
--target current --supervision joint --current-weight 100 --epsilon-weight .001
Use --physics-weight to enable scattered-field consistency; --tv-weight for TV.
--input-mode matrix predicts the entire image/volume without coordinates.
--channel 0 selects one incidence; --multi-input-mode separate treats each
incidence as a sample; default concatenates all available incidences.

2D labels/coordinates are column-major, as in the original project. 3D uses
the analogous column-major voxel order and a default measurement scale 1e5.
3D Fourier encoding fixes the original overlapping 4*i indices to 6*i.
For old 3D weights, explicitly use --legacy-3d-encoding. Existing raw NeRF
state_dicts (with or without module. prefixes) may be loaded with --weights;
set model/encoding/output options to match those original weights. New
checkpoints store their settings, so test.py only needs checkpoint and data.
Old two-channel epsilon heads are inferred automatically with --weights;
--epsilon-channels 2 also reproduces that head when training from scratch.
For a reproducibility run use --noise-mode legacy --seed 42. train.py records
the test noise-stream offset in its checkpoints. For standalone original
weights specify --legacy-noise-offset equal to the original training object
count. BatchNorm statistics depend on --point-chunk; a 32-object 64x64 batch
needs --point-chunk 131072 to train with the original whole-batch statistics.
The validated CYLINDER reproduction uses these overrides with train.py:
--dataset cylinder --epochs 200 --train-noise .30 --test-noise .30 --seed 42
--noise-mode legacy --epsilon-channels 2 --point-chunk 131072
(D=8, W=512, batch_size=32, Adam lr=1e-3, ten Fourier levels, no LR decay).

All paths resolve against the current working directory. Outputs belong in
an external run directory when uploading these Python source files to GitHub.
Original YAML files, plots, backups, pretrained weights and data are not needed.
"""

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys


@dataclass
class Config:
    dataset: str = "mnist"
    target: str = "epsilon"
    supervision: str = "joint"
    input_mode: str = "coordinate"
    multi_input_mode: str = "concat"
    channel: int = -1
    depth: int = 8
    width: int = 512
    skips: tuple = ()
    encoding_levels: int = 10
    epsilon_channels: int = 1
    legacy_3d_encoding: bool = False
    output_activation: str = "linear"
    output_scale: float = 1.0
    measurement_scale: float = 1.0
    batch_size: int = 32
    epochs: int = 300
    lr: float = 1e-3
    lr_step: int = 0
    lr_gamma: float = 0.9
    workers: int = 0
    point_chunk: int = 4096
    save_every: int = 20
    seed: int = 2
    train_noise: float = 0.05
    train_fraction: float = 1.0
    test_noise: float = 0.05
    noise_mode: str = "sample"
    legacy_noise_offset: int = 0
    epsilon_weight: float = 100.0
    current_weight: float = 100.0
    physics_weight: float = 0.0
    tv_weight: float = 0.0
    grad_clip: float = 0.0
    output_lower: float | None = None
    output_upper: float | None = None
    iou_threshold: float = 1.2
    wavelength: float | None = None
    domain_max: float | None = None
    filter_zero_epsilon: bool = False

    @property
    def dimensions(self):
        return 3 if self.dataset == "3d" else 2

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["skips"] = tuple(values.get("skips", ()))
        return cls(**values)

    def validate(self):
        if self.depth < 1 or self.width < 1 or self.batch_size < 1:
            raise ValueError("depth, width and batch_size must be positive")
        if self.encoding_levels < 0 or self.point_chunk < 1 or self.workers < 0 or self.epsilon_channels < 1:
            raise ValueError("Invalid encoding_levels, point_chunk or workers")
        if self.channel < -1 or self.epochs < 1 or self.save_every < 1:
            raise ValueError("Invalid channel, epochs or save_every")
        if self.legacy_noise_offset < 0:
            raise ValueError("legacy_noise_offset must be nonnegative")
        if not 0 < self.train_fraction <= 1:
            raise ValueError("train_fraction must be in (0, 1]")
        if self.lr <= 0 or self.lr_step < 0 or not 0 < self.lr_gamma <= 1:
            raise ValueError("Invalid learning-rate settings")
        if any(x < 0 for x in (self.train_noise, self.test_noise, self.epsilon_weight,
                               self.current_weight, self.physics_weight, self.tv_weight,
                               self.grad_clip)):
            raise ValueError("Noise, loss weights and clipping must be nonnegative")
        if any(i < 0 or i >= self.depth - 1 for i in self.skips):
            raise ValueError("skips indexes the depth-1 hidden layers")
        if self.target == "epsilon" and self.physics_weight:
            raise ValueError("Scattered-field physics loss requires --target current")
        if self.output_lower is not None and self.output_upper is not None:
            if self.output_lower >= self.output_upper:
                raise ValueError("output_lower must be smaller than output_upper")


def preset(dataset):
    conf = Config(dataset=dataset)
    if dataset == "if":
        return replace(conf, encoding_levels=40, train_noise=0.0, test_noise=0.0)
    if dataset == "3d":
        return replace(conf, width=256, batch_size=1, measurement_scale=1e5,
                       train_noise=0.0, test_noise=0.0, epsilon_weight=1.0,
                       wavelength=0.75, domain_max=1.0, filter_zero_epsilon=True)
    return conf


class ExperimentParser(argparse.ArgumentParser):
    """Translate a YAML train/test section into ordinary validated CLI flags."""

    def __init__(self, training, **kwargs):
        super().__init__(**kwargs)
        self.training = training

    def parse_args(self, args=None, namespace=None):
        argv = list(sys.argv[1:] if args is None else args)
        probe = argparse.ArgumentParser(add_help=False)
        probe.add_argument("--config")
        preliminary, _ = probe.parse_known_args(argv)
        prefix = []
        if preliminary.config and not any(x in argv for x in ("--help", "-h")):
            import yaml
            try:
                with Path(preliminary.config).open(encoding="utf-8") as handle:
                    payload = yaml.safe_load(handle)
                section = "train" if self.training else "test"
                if not isinstance(payload, dict) or section not in payload:
                    raise ValueError(f"YAML must contain a {section!r} section")
                if set(payload) - {"description", "train", "test"}:
                    raise ValueError("YAML accepts only description, train and test sections")
                values = payload[section]
                if not isinstance(values, dict):
                    raise ValueError(f"{section} must be a mapping of CLI option names to values")
                actions = {action.dest: action for action in self._actions}
                for key, value in values.items():
                    key = key.replace("-", "_")
                    if key not in actions or key in ("config", "help"):
                        raise ValueError(f"Unsupported {section} option: {key}")
                    if value is None:
                        continue
                    flag = "--" + key.replace("_", "-")
                    action = actions[key]
                    if isinstance(action, argparse.BooleanOptionalAction):
                        if not isinstance(value, bool):
                            raise ValueError(f"{key} must be a YAML boolean")
                        prefix.append(flag if value else "--no-" + key.replace("_", "-"))
                    else:
                        prefix.append(flag)
                        if isinstance(value, list):
                            prefix.extend(str(item) for item in value)
                        else:
                            prefix.append(str(value))
            except (OSError, ValueError, yaml.YAMLError) as error:
                self.error(str(error))
        # Explicit CLI options override YAML options. Normal argparse validation
        # still handles types, choices, required data paths and checkpoint flags.
        return super().parse_args(prefix + argv, namespace)


def parser(training):
    p = ExperimentParser(training, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="Paper YAML: use its train/test section; CLI options override it")
    p.add_argument("--dataset", type=str.lower, choices=("mnist", "cylinder", "if", "3d"),
                   default=None)
    if training:
        p.add_argument("--train-data", required=True)
        p.add_argument("--resume", help="Resume a unified checkpoint including optimizer")
    else:
        p.add_argument("--checkpoint", help="Checkpoint produced by train.py")
    p.add_argument("--test-data", required=not training)
    p.add_argument("--output", required=True, help="Run/results folder; preferably outside source folder")
    p.add_argument("--weights", help="Original NeRF raw state_dict; requires matching architecture settings")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    p.add_argument("--max-samples", type=int, help="Limit original objects for a short run")
    p.add_argument("--target", choices=("epsilon", "current"))
    p.add_argument("--supervision", choices=("current", "epsilon", "joint"))
    p.add_argument("--input-mode", choices=("coordinate", "matrix"))
    p.add_argument("--multi-input-mode", choices=("concat", "separate"))
    p.add_argument("--output-activation", choices=("linear", "tanh"))
    p.add_argument("--noise-mode", choices=("sample", "legacy"),
                   help="legacy reproduces the original NumPy RandomState noise stream")
    p.add_argument("--skips", type=int, nargs="*")
    for name in ("channel", "depth", "width", "encoding_levels", "batch_size", "epochs",
                 "lr_step", "workers", "point_chunk", "save_every", "seed",
                 "epsilon_channels", "legacy_noise_offset"):
        p.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("output_scale", "measurement_scale", "lr", "lr_gamma", "train_noise",
                 "test_noise", "epsilon_weight", "current_weight", "physics_weight",
                 "tv_weight", "grad_clip", "output_lower", "output_upper", "iou_threshold",
                 "wavelength", "domain_max", "train_fraction"):
        p.add_argument("--" + name.replace("_", "-"), type=float)
    for name in ("legacy_3d_encoding", "filter_zero_epsilon"):
        p.add_argument("--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction)
    return p


def configuration(args, saved=None):
    conf = Config.from_dict(saved) if saved is not None else preset(args.dataset or "mnist")
    if saved is None and args.target == "current":
        conf = replace(conf, output_activation="tanh",
                       output_scale=1e-3 if conf.dimensions == 3 else 1e-2,
                       epsilon_weight=0.001,
                       current_weight=1e7 if conf.dimensions == 3 else 100.0)
    overrides = {key: getattr(args, key) for key in conf.to_dict()
                 if getattr(args, key, None) is not None}
    if "skips" in overrides:
        overrides["skips"] = tuple(overrides["skips"])
    conf = replace(conf, **overrides)
    conf.validate()
    return conf
