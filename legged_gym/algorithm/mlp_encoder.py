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
from torch.nn.modules import rnn


class MLP_Encoder(nn.Module):
    is_mlp_encoder = True
    is_vae = False

    def __init__(
        self,
        num_input_dim,
        num_output_dim=None,              # 兼容旧签名（忽略）
        hidden_dims=[256, 256],
        activation="elu",
        orthogonal_init=False,
        output_detach=False,
        # 新增参数（可选，默认关闭两头模式以兼容旧权重）
        num_slots: int = 8,        # 槽位上限 L
        out_global_dim: int = 11,   # 全局头输出维度：默认 lin vel + mass + com(3) + inertia(3) + count(1)      
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        
        self.orthogonal_init = orthogonal_init
        self.output_detach = output_detach
        self.num_input_dim = int(num_input_dim)
        # self.num_output_dim = num_output_dim
        self.num_slots = int(num_slots)
        self.out_global_dim = int(out_global_dim)
        #激活函数选择​
        activation = get_activation(activation)

        # Encoder
        # backbone
        #第一层：输入层 → 第一个隐藏层，维度为 num_input_dim × hidden_dims[0]
        backbone_layers = []
        lin = nn.Linear(num_input_dim, hidden_dims[0])
        if self.orthogonal_init:
            torch.nn.init.orthogonal_(lin.weight, np.sqrt(2))
            torch.nn.init.constant_(lin.bias, 0.0)
        backbone_layers += [lin, activation]
        # 后续隐藏层
        for i in range(len(hidden_dims) - 1):
            lin = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            if self.orthogonal_init:
                torch.nn.init.orthogonal_(lin.weight, np.sqrt(2))
                torch.nn.init.constant_(lin.bias, 0.0)
            backbone_layers += [lin, activation]
        self.backbone = nn.Sequential(*backbone_layers)
        self.hidden_dim = hidden_dims[-1]
        
        
        # 全局头（11维）
        self.head_global = nn.Linear(self.hidden_dim, out_global_dim)
        if self.orthogonal_init:
            torch.nn.init.orthogonal_(self.head_global.weight, 0.01)
            torch.nn.init.constant_(self.head_global.bias, 0.0)
        # 槽位头（每个4维：[exist_logit, mass_raw, pos_x, pos_y]）
        self.head_slots = nn.ModuleList()
        for _ in range(num_slots):
            head = nn.Linear(self.hidden_dim, 4)  # [exist_logit, mass_raw, pos_x, pos_y]
            if self.orthogonal_init:
                torch.nn.init.orthogonal_(head.weight, 0.01)
                torch.nn.init.constant_(head.bias, 0.0)
            self.head_slots.append(head)  # 添加到 ModuleList
            
        # 对外声明输出维度，供对齐与构图
        self.num_output_dim = self.out_global_dim + 4 * self.num_slots
        # print(
        #     f"Encoder mode: {'two-heads' if self.use_two_heads else 'single-head'}, "
        #     f"backbone: {self.backbone}, "
        #     f"out_dim: {self.num_output_dim}"
        # )
        # 对外声明输出维度（P = 11 + 4L）
        self.num_output_dim = self.out_global_dim + 4 * self.num_slots
        # 缓存
        self.encoder_out = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 兼容 1D 输入：自动补 batch 维
        single_input = False
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, D]
            single_input = True
            
        z = self.backbone(x)                              # [B, H]
        g = self.head_global(z)                           # [B, out_global_dim]
        slots = [head(z) for head in self.head_slots]     # list of [B,4]
        s = torch.stack(slots, dim=1).reshape(z.size(0), -1)  # [B, 4L]
        out = torch.cat([g, s], dim=-1)                  # [B, out_global_dim + 4L]
        # 仅保留前三维，其余清零（调试用，完成后可删除）
        if out.shape[-1] > 3:
            out[:, 3:] = 0.0
        
        
        if single_input:
            out = out.squeeze(0)  # 还原 1D 以兼容旧调用
        return out
    

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.encoder_out = self.forward(x)
        return self.encoder_out.detach() if self.output_detach else self.encoder_out
    
    def get_encoder_out(self) -> torch.Tensor:
        return self.encoder_out
    
    def split_outputs(self, out: torch.Tensor = None):
        """将扁平输出拆成 (global_pred, slots_pred)。
        global_pred: [B, out_global_dim]
        slots_pred : [B, L, 4] -> [exist_logit, mass_raw, pos_x, pos_y]
        """
        if out is None:
            out = self.encoder_out
        B = out.size(0)
        g = out[:, :self.out_global_dim]
        s = out[:, self.out_global_dim:].view(B, self.num_slots, 4)
        return g, s

    def inference(self, input):
        with torch.no_grad():
            return self.encoder(input)

    # 便捷冻结/解冻接口（可选）
    def freeze_backbone(self, freeze: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
            
    def freeze_slot_head(self, idx: int, freeze: bool = True):
        """冻结/解冻指定槽位头(0-based)。"""
        assert 0 <= idx < self.num_slots, f"slot index out of range: {idx}"
        head = self.head_slots[idx]
        for p in head.parameters():
            p.requires_grad = not freeze
    
    def freeze_slot_heads(self, upto: int):
        """冻结前 upto 个槽位头(1-based),例如 upto=1 冻结槽1,保留 2..L 可训。"""
        upto = max(0, min(upto, self.num_slots))
        for i, head in enumerate(self.head_slots):
            freeze = (i < upto)
            for p in head.parameters():
                p.requires_grad = not freeze

                
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
