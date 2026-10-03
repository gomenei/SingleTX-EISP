"""Shared 2D pickle / 3D mmap adapters. No dependency on the original project.

E_s arrays: (objects, receivers, incidences), with a missing incidence axis
accepted for single-channel files. J: (objects, pixels, incidences[, 3]).
epsilon_gt and coordinates use NumPy column-major spatial flattening.
Each sample has measurements, epsilon, incidence indexes and optional current.
"""

from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def spatial_flatten(array):
    return np.asarray(array).reshape(-1, order="F")


class ScatteringDataset(Dataset):
    def __init__(self, path, conf, noise=0.0, max_samples=None, seed=None,
                 require_current_labels=True):
        self.path = Path(path).expanduser().resolve()
        self.conf = conf
        self.noise = noise
        self.seed = conf.seed if seed is None else seed
        self.dimensions = conf.dimensions
        self.components = 3 if self.dimensions == 3 else 1
        self.params = {}
        self.data = {}
        keys = ["E_s_real", "E_s_imag", "epsilon_gt", "x_dom", "y_dom"]
        if self.dimensions == 3:
            keys.append("z_dom")
        if conf.target == "current":
            keys += ["J_real", "J_imag", "E_inc_real", "E_inc_imag",
                     "Phi_mat_real", "Phi_mat_imag"]
            if conf.physics_weight:
                keys += ["R_mat_real", "R_mat_imag"]
        optional = ({"J_real", "J_imag"} if conf.target == "current" and not require_current_labels else set())
        if self.path.is_dir():
            for key in keys:
                file = self.path / (key + ".npy")
                if not file.is_file():
                    if key in optional:
                        continue
                    raise FileNotFoundError(f"Missing required data array: {file}")
                self.data[key] = np.load(file, mmap_mode="r", allow_pickle=False)
        else:
            if not self.path.is_file():
                raise FileNotFoundError(f"Data path does not exist: {self.path}")
            # The project's existing pickle format requires trusted local files.
            with self.path.open("rb") as handle:
                payload = pickle.load(handle)
            self.params = payload.get("params", {})
            raw = payload["data"]
            missing = set(keys) - set(raw) - optional
            if missing:
                raise ValueError(f"{self.path}: missing fields {sorted(missing)}")
            self.data = {key: as_numpy(raw[key]) for key in keys if key in raw}
        real, imag = self.data["E_s_real"], self.data["E_s_imag"]
        if real.shape != imag.shape or real.ndim not in (2, 3):
            raise ValueError("E_s real/imag must match and have shape (N,R[,I])")
        self.object_count, self.receivers = real.shape[:2]
        self.available_incidences = real.shape[2] if real.ndim == 3 else 1
        epsilon = self.data["epsilon_gt"]
        if epsilon.shape[0] != self.object_count:
            raise ValueError("E_s and epsilon_gt object counts differ")
        if epsilon.ndim == self.dimensions + 1:
            self.grid_shape = tuple(epsilon.shape[1:])
        elif epsilon.ndim == 2:
            coord_shape = self.data["x_dom"].shape
            if len(coord_shape) == self.dimensions:
                self.grid_shape = tuple(coord_shape)
            else:
                size = int(round(epsilon.shape[1] ** (1 / self.dimensions)))
                self.grid_shape = (size,) * self.dimensions
        else:
            raise ValueError("epsilon_gt must be (N,pixels) or (N,*spatial_shape)")
        self.pixel_count = int(np.prod(self.grid_shape))
        if int(np.prod(epsilon.shape[1:])) != self.pixel_count:
            raise ValueError("epsilon_gt size does not match the coordinate grid")
        coordinate_arrays = [spatial_flatten(self.data[key]) for key in keys
                             if key in ("x_dom", "y_dom", "z_dom")]
        if any(len(x) != self.pixel_count for x in coordinate_arrays):
            raise ValueError("Coordinate grids must contain one entry per pixel/voxel")
        self.coords = torch.from_numpy(np.stack(coordinate_arrays, axis=-1).astype(np.float32))
        if conf.channel >= self.available_incidences:
            # Single-channel training files are already selected; preserve the
            # physical incidence index for E_inc when params records it.
            if self.available_incidences != 1:
                raise ValueError(f"channel={conf.channel}; only {self.available_incidences} incidences available")
        self.selected = ([conf.channel] if conf.channel >= 0 else list(range(self.available_incidences)))
        self.incidence_count = (1 if conf.channel >= 0 or conf.multi_input_mode == "separate"
                                else self.available_incidences)
        self.feature_count = 2 * self.receivers * self.incidence_count
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be positive")
        count = min(self.object_count, max_samples) if max_samples else self.object_count
        self._legacy_noise = None
        if self.noise and conf.noise_mode == "legacy":
            rng = np.random.RandomState(self.seed)
            # Old loaders drew real then imag noise for each object in order.
            remaining = conf.legacy_noise_offset
            while remaining:
                block = min(remaining, 1024)
                rng.standard_normal((block, 2, self.receivers, self.available_incidences))
                remaining -= block
            self._legacy_noise = rng.standard_normal((count, 2, self.receivers, self.available_incidences))
        self.object_indices = []
        for index in range(count):
            values = epsilon[index]
            if not np.isfinite(values).all():
                continue
            if conf.filter_zero_epsilon and np.any(values == 0):
                continue
            self.object_indices.append(index)
        if not self.object_indices:
            raise ValueError("No valid samples remain after filtering")
        self.has_current = "J_real" in self.data and "J_imag" in self.data
        if ("J_real" in self.data) != ("J_imag" in self.data):
            raise ValueError("J_real and J_imag must either both exist or both be absent")
        if self.has_current:
            self._validate_currents()

    def _validate_currents(self):
        for key in ("J_real", "J_imag"):
            arr = self.data[key]
            if arr.shape[:2] != (self.object_count, self.pixel_count):
                raise ValueError(f"{key} must start with (objects,pixels)")
            expected = (self.object_count, self.pixel_count, self.available_incidences, self.components)
            if self.dimensions == 2 and arr.ndim == 2:
                shape = (*arr.shape, 1, 1)
            elif self.dimensions == 2 and arr.ndim == 3:
                shape = (*arr.shape, 1)
            elif self.dimensions == 3 and arr.ndim == 3 and self.available_incidences == 1:
                shape = (arr.shape[0], arr.shape[1], 1, arr.shape[2])
            else:
                shape = arr.shape
            if tuple(shape) != expected:
                raise ValueError(f"{key}: expected {expected}, got {arr.shape}")

    def __len__(self):
        repeats = self.available_incidences if self.conf.channel == -1 and self.conf.multi_input_mode == "separate" else 1
        return len(self.object_indices) * repeats

    def __getitem__(self, index):
        separate = self.conf.channel == -1 and self.conf.multi_input_mode == "separate"
        repeats = self.available_incidences if separate else 1
        obj = self.object_indices[index // repeats]
        physical = [index % repeats] if separate else self.selected
        local = [0] if self.available_incidences == 1 else physical
        real = np.array(self.data["E_s_real"][obj], dtype=np.float32, copy=True).reshape(self.receivers, -1)
        imag = np.array(self.data["E_s_imag"][obj], dtype=np.float32, copy=True).reshape(self.receivers, -1)
        if self.noise:
            if self._legacy_noise is not None:
                # Match the old torch energy -> float64 NumPy perturbation ->
                # float32 assignment, including its operation order.
                energy = torch.sqrt(torch.mean(torch.from_numpy(real) ** 2 + torch.from_numpy(imag) ** 2))
                energy = energy * (1 / torch.sqrt(torch.tensor([2])))
                scale = float((energy * self.noise)[0])
                real[:] = real.astype(np.float64) + scale * self._legacy_noise[obj, 0]
                imag[:] = imag.astype(np.float64) + scale * self._legacy_noise[obj, 1]
            else:
                rng = np.random.default_rng(self.seed + obj)
                scale = np.sqrt(np.mean(real ** 2 + imag ** 2) / 2) * self.noise
                real += scale * rng.standard_normal(real.shape).astype(np.float32)
                imag += scale * rng.standard_normal(imag.shape).astype(np.float32)
        real, imag = real[:, local], imag[:, local]
        if self.dimensions == 2:
            # incident-major: each incident has all real then all imag receivers.
            measurements = np.concatenate((real, imag), axis=0).T.reshape(-1)
        else:
            # Retain original 3D receiver-major: real incidences, imag incidences.
            measurements = np.concatenate((real, imag), axis=1).reshape(-1)
        item = {
            "measurements": torch.from_numpy(measurements.copy()),
            "epsilon": torch.from_numpy(spatial_flatten(self.data["epsilon_gt"][obj]).astype(np.float32)),
            "incidences": torch.tensor(physical, dtype=torch.long),
            "object_index": torch.tensor(obj, dtype=torch.long),
        }
        if self.has_current:
            parts = [np.asarray(self.data[key][obj]).reshape(self.pixel_count, self.available_incidences,
                                                           self.components)[:, local, :]
                     for key in ("J_real", "J_imag")]
            item["current"] = torch.from_numpy(np.stack(parts, axis=-1).astype(np.float32))
        return item

    def signature(self):
        return dict(dimensions=self.dimensions, grid_shape=list(self.grid_shape),
                    receivers=self.receivers, incidence_count=self.incidence_count,
                    feature_count=self.feature_count, components=self.components)
