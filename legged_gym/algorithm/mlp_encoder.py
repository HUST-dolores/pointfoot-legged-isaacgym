# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


def build_mlp(
    input_dim,
    output_dim,
    hidden_dims,
    activation_name,
    orthogonal_init=False,
    output_scale=0.01,
):
    layers = []
    if len(hidden_dims) == 0:
        layers.append(nn.Linear(input_dim, output_dim))
        if orthogonal_init:
            torch.nn.init.orthogonal_(layers[-1].weight, output_scale)
            torch.nn.init.constant_(layers[-1].bias, 0.0)
        return nn.Sequential(*layers)

    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    if orthogonal_init:
        torch.nn.init.orthogonal_(layers[-1].weight, np.sqrt(2))
        torch.nn.init.constant_(layers[-1].bias, 0.0)
    layers.append(get_activation(activation_name))

    for l in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
        if orthogonal_init:
            torch.nn.init.orthogonal_(layers[-1].weight, np.sqrt(2))
            torch.nn.init.constant_(layers[-1].bias, 0.0)
        layers.append(get_activation(activation_name))

    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    if orthogonal_init:
        torch.nn.init.orthogonal_(layers[-1].weight, output_scale)
        torch.nn.init.constant_(layers[-1].bias, 0.0)
    return nn.Sequential(*layers)


class SharedBackboneDualHead(nn.Module):
    def __init__(
        self,
        num_input_dim,
        backbone_hidden_dims,
        activation,
        orthogonal_init,
        vel_head_hidden_dims,
        mass_head_hidden_dims,
    ):
        super().__init__()

        if len(backbone_hidden_dims) == 0:
            raise ValueError("backbone hidden_dims must contain at least 1 layer")

        self.backbone = build_mlp(
            num_input_dim,
            backbone_hidden_dims[-1],
            backbone_hidden_dims,
            activation,
            orthogonal_init=orthogonal_init,
            output_scale=np.sqrt(2),
        )
        trunk_dim = backbone_hidden_dims[-1]
        self.vel_head = build_mlp(
            trunk_dim,
            3,
            vel_head_hidden_dims,
            activation,
            orthogonal_init=orthogonal_init,
            output_scale=0.01,
        )
        self.mass_head = build_mlp(
            trunk_dim,
            4,
            mass_head_hidden_dims,
            activation,
            orthogonal_init=orthogonal_init,
            output_scale=0.01,
        )

    def forward(self, input):
        trunk = self.backbone(input)
        vel_out = self.vel_head(trunk)
        mass_out = self.mass_head(trunk)
        return torch.cat((vel_out, mass_out), dim=-1)


class LegacySingleHead(nn.Module):
    def __init__(
        self,
        num_input_dim,
        num_output_dim,
        hidden_dims,
        activation,
        orthogonal_init,
    ):
        super().__init__()
        self.net = build_mlp(
            num_input_dim,
            num_output_dim,
            hidden_dims,
            activation,
            orthogonal_init=orthogonal_init,
            output_scale=0.01,
        )

    def forward(self, input):
        return self.net(input)


class MLP_Encoder(nn.Module):
    is_mlp_encoder = True
    is_vae = False

    def __init__(
        self,
        num_input_dim,
        num_output_dim,
        hidden_dims=[256, 256], #之前我修改的是512256128 修改
        use_dual_head=False,
        vel_head_hidden_dims=[],
        mass_head_hidden_dims=[],
        activation="elu",
        orthogonal_init=False,
        output_detach=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(MLP_Encoder, self).__init__()

        self.orthogonal_init = orthogonal_init
        self.output_detach = output_detach
        self.num_input_dim = num_input_dim
        self.num_output_dim = num_output_dim
        self.use_dual_head = use_dual_head
        self.vel_head_out_dim = 3
        self.mass_head_out_dim = 4

        if self.use_dual_head and self.num_output_dim != (
            self.vel_head_out_dim + self.mass_head_out_dim
        ):
            raise ValueError(
                "Dual-head encoder requires num_output_dim == 7 "
                f"(got {self.num_output_dim})."
            )

        if self.use_dual_head:
            self.encoder = SharedBackboneDualHead(
                num_input_dim,
                hidden_dims,
                activation,
                self.orthogonal_init,
                vel_head_hidden_dims,
                mass_head_hidden_dims,
            )
        else:
            self.encoder = LegacySingleHead(
                num_input_dim,
                num_output_dim,
                hidden_dims,
                activation,
                self.orthogonal_init,
            )

        print(f"Encoder MLP: {self.encoder}")

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def forward(self, input):
        return self.encoder(input)

    def encode(self, input):
        self.encoder_out = self.encoder(input)
        if self.output_detach:
            return self.encoder_out.detach()
        else:
            return self.encoder_out

    def get_encoder_out(self):
        return self.encoder_out

    def inference(self, input):
        with torch.no_grad():
            return self.encoder(input)
