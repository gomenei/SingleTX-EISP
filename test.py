"""Standalone evaluation requiring only test data and a checkpoint.

Results: metrics.json and compressed per-batch predictions in predictions/.
eps_pred/eps_true are flattened column-major; reconstruct one image with
array.reshape(grid_shape, order='F'). See config.py for full usage.
"""

from pathlib import Path

from config import configuration, parser
from dataset import ScatteringDataset
from engine import (Runner, check_signature, choose_device, legacy_output_config, load_weights, make_loader,
                    read_checkpoint, set_seed, write_json)
from model import NeRF, model_spec
from physics import Physics


def main(argv=None):
    p = parser(training=False)
    args = p.parse_args(argv)
    if bool(args.checkpoint) == bool(args.weights):
        p.error("Supply exactly one of --checkpoint (unified) or --weights (original raw state_dict)")
    payload = read_checkpoint(args.checkpoint or args.weights)
    if args.checkpoint and "config" not in payload:
        p.error("This is a raw state_dict: use --weights and matching model options")
    saved_config = payload.get("evaluation_config", payload.get("config")) if args.checkpoint else None
    conf = configuration(args, saved_config)
    if args.weights:
        conf = legacy_output_config(conf, payload, args.epsilon_channels)
    set_seed(conf.seed)
    device = choose_device(args.device)
    data = ScatteringDataset(args.test_data, conf, conf.test_noise, args.max_samples,
                             seed=conf.seed if conf.noise_mode == "legacy" else conf.seed + 1_000_000,
                             require_current_labels=False)
    spec = model_spec(data, conf)
    if args.checkpoint:
        check_signature(data, payload["data_signature"])
        if spec != payload["model_spec"]:
            raise ValueError("Model settings differ from the checkpoint")
        # Encoding/data options may alter behavior without altering layer sizes.
        saved = payload["config"]
        for key in ("target", "input_mode", "multi_input_mode", "channel",
                    "encoding_levels", "legacy_3d_encoding", "measurement_scale"):
            if getattr(conf, key) != saved[key]:
                raise ValueError(f"--{key.replace('_', '-')} must match checkpoint: {saved[key]}")
    model = NeRF(**spec).to(device)
    load_weights(model, payload)
    physics = Physics(data, conf, device) if conf.target == "current" else None
    runner = Runner(model, data, conf, device, physics)
    output = Path(args.output).expanduser().resolve()
    if (output / "metrics.json").exists() or (output / "predictions").exists():
        raise FileExistsError("Evaluation output already exists; choose a new results folder")
    loader = make_loader(data, conf, device)
    scores = runner.run(loader, prediction_dir=output / "predictions")
    write_json(output / "metrics.json", dict(metrics=scores, config=conf.to_dict(),
               checkpoint=str(Path(args.checkpoint or args.weights).resolve()),
               data=str(data.path), data_signature=data.signature()))
    print(scores)
    print(f"Saved metrics and predictions to {output}")


if __name__ == "__main__":
    main()
