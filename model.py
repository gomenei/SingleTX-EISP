"""NeRF-style coordinate MLP, retaining the original layer/state_dict names."""

import torch
from torch import nn
from torch.nn import functional as F


def encode_coordinates(coords, levels, legacy_3d=False):
    if not levels:
        return coords
    dims = coords.shape[-1]
    encoded = coords.new_zeros((coords.shape[0], 2 * dims * levels))
    for level in range(levels):
        offset = (4 if legacy_3d and dims == 3 else 2 * dims) * level
        for axis in range(dims):
            values = coords[:, axis] * (2.0 ** level)
            encoded[:, offset + 2 * axis] = values.sin()
            encoded[:, offset + 2 * axis + 1] = values.cos()
    return encoded


class NeRF(nn.Module):
    def __init__(self, input_ch, output_ch, depth=8, width=512, skips=(),
                 output_activation="linear", output_scale=1.0):
        super().__init__()
        self.skips = tuple(skips)
        self.output_activation = output_activation
        self.output_scale = output_scale
        self.input_layer = nn.Linear(input_ch, width)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(width + input_ch if i in self.skips else width, width)
            for i in range(depth - 1)
        ])
        self.bn1 = nn.BatchNorm1d(width)
        self.output_layer = nn.Linear(width, output_ch)

    def forward(self, inputs):
        x = F.relu(self.input_layer(inputs))
        for i, layer in enumerate(self.hidden_layers):
            if i in self.skips:
                x = torch.cat((inputs, x), dim=-1)
            x = layer(x)
            if i == 4:
                # BatchNorm cannot estimate variance for a single matrix sample.
                if self.training and x.shape[0] == 1:
                    x = F.batch_norm(x, self.bn1.running_mean, self.bn1.running_var,
                                     self.bn1.weight, self.bn1.bias, training=False,
                                     eps=self.bn1.eps)
                else:
                    x = self.bn1(x)
            x = F.relu(x)
        output = self.output_layer(x)
        if self.output_activation == "tanh":
            output = torch.tanh(output)
        return self.output_scale * output


def model_spec(dataset, conf):
    coordinates = encode_coordinates(dataset.coords, conf.encoding_levels,
                                     conf.legacy_3d_encoding)
    input_ch = dataset.feature_count
    point_outputs = conf.epsilon_channels if conf.target == "epsilon" else 2 * dataset.incidence_count * dataset.components
    if conf.input_mode == "coordinate":
        input_ch += coordinates.shape[-1]
        output_ch = point_outputs
    else:
        output_ch = point_outputs * dataset.pixel_count
    return dict(input_ch=input_ch, output_ch=output_ch, depth=conf.depth,
                width=conf.width, skips=conf.skips,
                output_activation=conf.output_activation, output_scale=conf.output_scale)
