"""Differentiable current -> permittivity / scattered-field conversions."""

import numpy as np
import torch


class Physics:
    def __init__(self, dataset, conf, device):
        self.dimensions = dataset.dimensions
        self.receivers = dataset.receivers
        self.pixel_count = dataset.pixel_count
        self.components = dataset.components
        self.e_inc = self._complex(dataset, "E_inc", device)
        if self.dimensions == 2:
            self.e_inc = self.e_inc.reshape(self.pixel_count, -1, 1)
        elif self.e_inc.ndim == 2:
            self.e_inc = self.e_inc.reshape(self.pixel_count, 1, self.components)
        if self.e_inc.shape[0] != self.pixel_count or self.e_inc.shape[-1] != self.components:
            raise ValueError("E_inc must have shape (pixels,incidences[,components])")
        self.phi = self._complex(dataset, "Phi_mat", device)
        if tuple(self.phi.shape) != (self.pixel_count, self.pixel_count):
            raise ValueError("Phi_mat must have shape (pixels,pixels)")
        self.r_mat = None
        if conf.physics_weight:
            self.r_mat = self._complex(dataset, "R_mat", device)
            if self.dimensions == 2:
                if self.r_mat.ndim == 2:
                    self.r_mat = self.r_mat.unsqueeze(0)
                # IF stores one receiver operator per physical incidence.
                self.r_mat = self.r_mat.unsqueeze(-1)
            elif self.r_mat.ndim == 3:
                self.r_mat = self.r_mat.unsqueeze(0)
            if tuple(self.r_mat.shape[1:]) != (self.receivers, self.pixel_count, self.components):
                raise ValueError("R_mat must be (receivers,pixels[,components]) or (incidences,receivers,pixels[,components])")
        wavelength = conf.wavelength if conf.wavelength is not None else dataset.params.get("lam_0")
        if wavelength is None or float(wavelength) <= 0:
            raise ValueError("Current reconstruction needs a positive lam_0 or --wavelength")
        extent = conf.domain_max if conf.domain_max is not None else dataset.params.get("MAX")
        if extent is None or float(extent) <= 0:
            raise ValueError("Current reconstruction needs positive MAX or --domain-max")
        if len(set(dataset.grid_shape)) != 1 or dataset.grid_shape[0] < 2:
            raise ValueError("Current physics currently requires a square/cubic uniform grid")
        cell = (2 * float(extent) / (dataset.grid_shape[0] - 1)) ** self.dimensions
        omega = 2 * np.pi * 3e8 / float(wavelength)
        self.factor = omega * 8.85e-12 * cell

    @staticmethod
    def _complex(dataset, name, device):
        real = torch.from_numpy(np.array(dataset.data[name + "_real"], dtype=np.float32, copy=True))
        imag = torch.from_numpy(np.array(dataset.data[name + "_imag"], dtype=np.float32, copy=True))
        return torch.complex(real, imag).to(device)

    def dielectric(self, current, incidences):
        """current: (B,P,K,C,2), with real/imag in the last axis."""
        values = torch.complex(current[..., 0], current[..., 1])
        if self.e_inc.shape[1] == 1:
            indexes = torch.zeros_like(incidences)
        else:
            indexes = incidences
            if torch.any(indexes >= self.e_inc.shape[1]):
                raise ValueError("Selected physical incidence does not exist in E_inc")
        incident = self.e_inc[:, indexes, :].permute(1, 0, 2, 3)
        total = incident + torch.einsum("pq,bqkc->bpkc", self.phi, values)
        total = torch.where(total.abs() < 1e-12, total + 1e-12, total)
        return (1.0 - (values / total).imag / self.factor).mean(dim=(2, 3))

    def scattered(self, current, incidences=None):
        if self.r_mat is None:
            raise RuntimeError("R_mat was not loaded; enable physics_weight")
        values = torch.complex(current[..., 0], current[..., 1])
        if self.r_mat.shape[0] == 1:
            fields = torch.einsum("rpc,bpkc->brk", self.r_mat[0], values)
            if self.dimensions == 2:
                return torch.cat((fields.real, fields.imag), dim=1).transpose(1, 2).flatten(1)
            return torch.cat((fields.real, fields.imag), dim=2).flatten(1)
        if incidences is None:
            incidences = torch.arange(values.shape[2], device=values.device).expand(values.shape[0], -1)
        indexes = incidences
        if torch.any(indexes >= self.r_mat.shape[0]):
            raise ValueError("Selected physical incidence does not exist in R_mat")
        fields = torch.einsum("bkrpc,bpkc->brk", self.r_mat[indexes], values)
        if self.dimensions == 2:
            return torch.cat((fields.real, fields.imag), dim=1).transpose(1, 2).flatten(1)
        return torch.cat((fields.real, fields.imag), dim=2).flatten(1)
