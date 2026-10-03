"""Shared train/evaluate, chunked coordinate prediction and checkpoint IO."""

import json
from dataclasses import replace
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from metrics import Metrics, finite_json
from model import encode_coordinates


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def make_loader(dataset, conf, device, shuffle=False):
    return DataLoader(dataset, batch_size=conf.batch_size, shuffle=shuffle,
                      num_workers=conf.workers, pin_memory=device.type == "cuda")


def read_checkpoint(path, device="cpu"):
    # New checkpoints contain tensors and simple dict/list/scalar metadata.
    return torch.load(Path(path), map_location=device, weights_only=True)


def load_weights(model, payload):
    state = payload.get("model_state", payload.get("state_dict", payload))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def legacy_output_config(conf, payload, explicit_channels=None):
    """Old 2D epsilon networks emitted two values and used only channel zero."""
    if conf.target != "epsilon" or conf.input_mode != "coordinate" or explicit_channels is not None:
        return conf
    state = payload.get("model_state", payload.get("state_dict", payload))
    key = next((key for key in state if key.removeprefix("module.") == "output_layer.weight"), None)
    if key is not None:
        return replace(conf, epsilon_channels=state[key].shape[0])
    return conf


def check_signature(dataset, expected):
    actual = dataset.signature()
    if actual != expected:
        raise ValueError(f"Data/model layout mismatch. Checkpoint: {expected}; data: {actual}")


def write_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(finite_json(values), handle, ensure_ascii=False, indent=2, allow_nan=False)


class Runner:
    def __init__(self, model, dataset, conf, device, physics=None):
        self.model = model
        self.conf = conf
        self.device = device
        self.physics = physics
        self.grid_shape = dataset.grid_shape
        self.pixel_count = dataset.pixel_count
        self.incidences = dataset.incidence_count
        self.components = dataset.components
        self.coords = encode_coordinates(dataset.coords.to(device), conf.encoding_levels,
                                         conf.legacy_3d_encoding)

    def clip_epsilon(self, values):
        if self.conf.output_lower is not None or self.conf.output_upper is not None:
            return values.clamp(min=self.conf.output_lower, max=self.conf.output_upper)
        return values

    def feature_chunks(self, measurements):
        """Build only a chunk of the repeated field/coordinate input at once."""
        batch = measurements.shape[0]
        total = batch * self.pixel_count
        scaled = measurements * self.conf.measurement_scale
        for start in range(0, total, self.conf.point_chunk):
            stop = min(start + self.conf.point_chunk, total)
            indexes = torch.arange(start, stop, device=self.device)
            features = torch.cat((scaled[indexes // self.pixel_count],
                                  self.coords[indexes % self.pixel_count]), dim=1)
            yield start, stop, features

    def predict(self, measurements):
        batch = measurements.shape[0]
        if self.conf.input_mode == "matrix":
            raw = self.model(measurements * self.conf.measurement_scale)
        else:
            raw = torch.cat([self.model(features) for _, _, features in self.feature_chunks(measurements)])
        if self.conf.target == "epsilon":
            return self.clip_epsilon(raw.reshape(batch, self.pixel_count, self.conf.epsilon_channels)[..., 0])
        # Original point output: all real channels/components, then all imag.
        raw = raw.reshape(batch, self.pixel_count, 2, self.incidences, self.components)
        return raw.permute(0, 1, 3, 4, 2).contiguous()

    def losses(self, predicted, batch):
        zero = predicted.new_zeros(())
        current_loss = physics_loss = zero
        if self.conf.target == "current":
            if self.physics is None:
                raise RuntimeError("Current prediction requires physics metadata")
            epsilon = self.clip_epsilon(self.physics.dielectric(predicted, batch["incidences"]))
            if self.conf.supervision in ("current", "joint") and "current" in batch:
                current_loss = self.conf.current_weight * F.mse_loss(predicted, batch["current"])
            if self.conf.physics_weight:
                physics_loss = self.conf.physics_weight * F.mse_loss(
                    self.physics.scattered(predicted, batch["incidences"]), batch["measurements"])
        else:
            epsilon = predicted
        epsilon_loss = zero
        if self.conf.target == "epsilon" or self.conf.supervision in ("epsilon", "joint") or "current" not in batch:
            epsilon_loss = self.conf.epsilon_weight * F.mse_loss(epsilon, batch["epsilon"])
        tv_loss = zero
        if self.conf.tv_weight:
            # Column-major image/volume reshaped with reversed spatial axes.
            volume = epsilon.reshape(epsilon.shape[0], *reversed(self.grid_shape))
            for axis in range(1, volume.ndim):
                if volume.shape[axis] > 1:
                    tv_loss = tv_loss + torch.diff(volume, dim=axis).square().mean()
            tv_loss = self.conf.tv_weight * tv_loss
        loss = epsilon_loss + current_loss + physics_loss + tv_loss
        return loss, epsilon

    def _streamed_epsilon_train(self, batch):
        # Independent epsilon MSE lets us backward each chunk immediately,
        # bounding activation memory even for a complete 3D volume.
        truth = batch["epsilon"].flatten()
        predictions, total = [], 0.0
        for start, stop, features in self.feature_chunks(batch["measurements"]):
            prediction = self.clip_epsilon(self.model(features)[:, 0])
            loss = self.conf.epsilon_weight * F.mse_loss(prediction, truth[start:stop], reduction="sum") / truth.numel()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; inspect data/scales")
            loss.backward()
            total += loss.detach().item()
            predictions.append(prediction.detach())
        return total, torch.cat(predictions).reshape_as(batch["epsilon"])

    def run(self, loader, optimizer=None, prediction_dir=None):
        training = optimizer is not None
        self.model.train(training)
        metrics = Metrics(self.grid_shape, self.conf.iou_threshold)
        loss_total = 0.0
        sample_count = 0
        if prediction_dir is not None:
            prediction_dir = Path(prediction_dir)
            prediction_dir.mkdir(parents=True, exist_ok=True)
        with torch.set_grad_enabled(training):
            for batch_index, raw_batch in enumerate(loader):
                batch = {key: value.to(self.device, non_blocking=True) for key, value in raw_batch.items()}
                if training:
                    optimizer.zero_grad(set_to_none=True)
                streamed = (training and self.conf.target == "epsilon"
                            and self.conf.input_mode == "coordinate" and not self.conf.tv_weight)
                if streamed:
                    loss_value, epsilon = self._streamed_epsilon_train(batch)
                else:
                    prediction = self.predict(batch["measurements"])
                    loss, epsilon = self.losses(prediction, batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite loss; inspect data/scales")
                    loss_value = loss.detach().item()
                    if training:
                        loss.backward()
                if not torch.isfinite(epsilon).all():
                    raise FloatingPointError("Non-finite permittivity reconstruction")
                if training:
                    if self.conf.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.grad_clip)
                    optimizer.step()
                scores = metrics.update(epsilon, batch["epsilon"])
                count = batch["epsilon"].shape[0]
                loss_total += loss_value * count
                sample_count += count
                if prediction_dir is not None:
                    # Bounded-memory results: one compressed file per batch.
                    values = dict(eps_pred=epsilon.detach().cpu().numpy(),
                                  eps_true=batch["epsilon"].cpu().numpy(),
                                  object_index=batch["object_index"].cpu().numpy(),
                                  incidences=batch["incidences"].cpu().numpy(),
                                  grid_shape=np.array(self.grid_shape),
                                  flatten_order=np.array("F"), **scores)
                    if self.conf.target == "current":
                        values["current_pred"] = prediction.detach().cpu().numpy()
                    np.savez_compressed(prediction_dir / f"batch_{batch_index:06d}.npz", **values)
        result = metrics.result()
        result["loss"] = loss_total / sample_count
        result["samples"] = sample_count
        return result
