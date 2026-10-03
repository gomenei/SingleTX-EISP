"""Train MNIST, CYLINDER, IF or 3D with the shared pipeline; see config.py."""

import json
from dataclasses import replace
from pathlib import Path

import torch

from config import configuration, parser
from dataset import ScatteringDataset
from engine import (Runner, check_signature, choose_device, legacy_output_config, load_weights, make_loader,
                    read_checkpoint, set_seed, write_json)
from model import NeRF, model_spec
from metrics import finite_json
from physics import Physics


def main(argv=None):
    args = parser(training=True).parse_args(argv)
    if args.resume and args.weights:
        raise ValueError("Use either --resume or --weights")
    checkpoint = read_checkpoint(args.resume) if args.resume else None
    conf = configuration(args, checkpoint["config"] if checkpoint else None)
    initial_weights = read_checkpoint(args.weights) if args.weights else None
    if initial_weights is not None:
        conf = legacy_output_config(conf, initial_weights, args.epsilon_channels)
    set_seed(conf.seed)
    device = choose_device(args.device)
    train_data = ScatteringDataset(args.train_data, conf, conf.train_noise, args.max_samples,
                                   require_current_labels=(conf.target == "current" and
                                                           conf.supervision in ("current", "joint")))
    # Paper Table 3 reduces training objects, never the evaluation set.
    # A private generator keeps model initialization and loader RNG unchanged.
    valid_train_count = len(train_data.object_indices)
    if conf.train_fraction < 1:
        count = max(1, int(valid_train_count * conf.train_fraction))
        generator = torch.Generator().manual_seed(conf.seed)
        chosen = torch.randperm(valid_train_count, generator=generator)[:count].sort().values.tolist()
        train_data.object_indices = [train_data.object_indices[i] for i in chosen]
    spec = model_spec(train_data, conf)
    if checkpoint:
        check_signature(train_data, checkpoint["data_signature"])
        if spec != checkpoint["model_spec"]:
            raise ValueError("Resume architecture differs from the checkpoint")
    model = NeRF(**spec).to(device)
    if checkpoint:
        load_weights(model, checkpoint)
    elif args.weights:
        load_weights(model, initial_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=conf.lr_step,
                                                gamma=conf.lr_gamma) if conf.lr_step else None)
    start_epoch, best_loss = 0, float("inf")
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if (scheduler is None) != (checkpoint.get("scheduler_state") is None):
            raise ValueError("Resume must use the checkpoint's scheduler settings")
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    if start_epoch >= conf.epochs:
        raise ValueError("Resume already reached --epochs; increase the total epoch count")
    train_physics = Physics(train_data, conf, device) if conf.target == "current" else None
    train_runner = Runner(model, train_data, conf, device, train_physics)
    train_loader = make_loader(train_data, conf, device, shuffle=True)
    test_runner = test_loader = None
    test_conf = conf
    if args.test_data:
        if conf.noise_mode == "legacy":
            test_conf = replace(conf, legacy_noise_offset=conf.legacy_noise_offset + train_data.object_count)
        test_seed = conf.seed if conf.noise_mode == "legacy" else conf.seed + 1_000_000
        test_data = ScatteringDataset(args.test_data, test_conf, conf.test_noise, args.max_samples,
                                      seed=test_seed, require_current_labels=False)
        check_signature(test_data, train_data.signature())
        # Every split uses its own forward operator and incident fields.
        test_physics = Physics(test_data, conf, device) if conf.target == "current" else None
        test_runner = Runner(model, test_data, test_conf, device, test_physics)
        test_loader = make_loader(test_data, conf, device)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((output / name).exists() for name in ("last.pt", "best.pt", "history.jsonl")):
        raise FileExistsError("Output already contains a run; choose a new folder or use --resume")
    write_json(output / "config.json", dict(config=conf.to_dict(), model_spec=spec,
               data_signature=train_data.signature(), train_data=str(train_data.path),
               train_subset=dict(fraction=conf.train_fraction, valid_objects=valid_train_count,
                                 selected_objects=len(train_data.object_indices),
                                 strategy="seeded random selection in original order"),
               test_data=str(test_data.path) if args.test_data else None))
    print(f"device={device}; train_samples={len(train_data)}; layout={train_data.signature()}")
    print("best.pt selection: evaluation loss" if test_loader else "best.pt selection: training loss (no test data)")
    for epoch in range(start_epoch, conf.epochs):
        train_scores = train_runner.run(train_loader, optimizer=optimizer)
        test_scores = test_runner.run(test_loader) if test_loader else None
        selected_loss = (test_scores or train_scores)["loss"]
        improved = selected_loss < best_loss
        best_loss = min(best_loss, selected_loss)
        if scheduler:
            scheduler.step()
        record = dict(epoch=epoch + 1, lr=optimizer.param_groups[0]["lr"],
                      train=train_scores, test=test_scores)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(finite_json(record), ensure_ascii=False, allow_nan=False) + "\n")
        print(f"epoch {epoch + 1}/{conf.epochs}: train_loss={train_scores['loss']:.6g}"
              + (f" test_loss={test_scores['loss']:.6g} PSNR={test_scores['psnr']:.3f}" if test_scores else ""), flush=True)
        payload = dict(format_version=1, epoch=epoch + 1, best_loss=best_loss,
                       config=conf.to_dict(), evaluation_config=test_conf.to_dict(), model_spec=spec,
                       data_signature=train_data.signature(),
                       model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                       scheduler_state=scheduler.state_dict() if scheduler else None,
                       torch_rng_state=torch.get_rng_state(),
                       cuda_rng_state=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
        torch.save(payload, output / "last.pt")
        if improved:
            torch.save(payload, output / "best.pt")
        if (epoch + 1) % conf.save_every == 0:
            torch.save(payload, output / f"epoch_{epoch + 1:04d}.pt")
    write_json(output / "metrics.json", record)
    print(f"Saved checkpoints and metrics to {output}")


if __name__ == "__main__":
    main()
