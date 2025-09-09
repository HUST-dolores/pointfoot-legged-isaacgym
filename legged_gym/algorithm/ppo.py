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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .mlp_encoder import MLP_Encoder
from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage


class PPO:
    actor_critic: ActorCritic
    encoder: MLP_Encoder

    def __init__(
        self,
        num_group,
        encoder,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        vae_beta=1.0,
        est_learning_rate=1.0e-3,
        ts_learning_rate=1.0e-4,
        critic_take_latent=False,
        early_stop=False,
        anneal_lr=False,
        device="cpu",
    ):
        self.device = device
        self.num_group = num_group

        self.desired_kl = desired_kl
        self.early_stop = early_stop
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.vae_beta = vae_beta
        self.critic_take_latent = critic_take_latent

        self.encoder = encoder
        self.encoder.to(self.device)  # 放到同一 device

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam([{"params": self.actor_critic.parameters()}], lr=learning_rate)

        if self.encoder.num_output_dim != 0:
            self.extra_optimizer = optim.Adam(
                self.encoder.parameters(), lr=est_learning_rate
            )
        else:
            self.extra_optimizer = None
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        #encoder 自适应冻结配置
                # 冻结/门控超参（默认值，可在后续接 cfg 覆盖）
        self.enc_freeze_cfg = {
            "p_thresh": 0.7,          # 槽存在概率阈值
            "ema_beta": 0.9,          # EMA 衰减
            "patience": 5,            # 连续满足步数
            "min_updates": 50,        # 冻结前最少更新步
            # 仅用“输出稳定性判据”时的阈值
            "mass_var_thresh": 1e-2,  # 质量时间方差阈值
            "pos_var_thresh": 1e-3,   # 位置时间方差阈值
            # 若改回“特权误差判据”，也给默认
            "mass_mse_thresh": 1e-2,
            "pos_mse_thresh": 5e-4,
            # 解冻（被拿走）判据
            "unlock_p_thresh": 0.3,     # 冻结槽位 p 低于此阈值
            "unlock_patience": 10,      # 连续步数
            "alpha_mass": 1.0,          # 质量匹配权重
            "beta_com": 1.0,            # 质心匹配权重
            # 额外：全局 g_pred 的索引与一致性判据
            "g_mass_idx": 3,            # g_pred 中总质量索引
            "g_com_start": 4,           # g_pred 中质心起始索引
            "g_com_dim": 2,             # 质心维度(2: x,y)
            "mass_margin_frac": 0.20,   # 允许的质量相对误差
            "com_margin": 0.05,         # 允许的质心误差(m)
            "unlock_by_consistency": True,  # 是否启用一致性解冻 
        }
        
        
        L = self.encoder.num_slots
        self.enc_slot_frozen = [False] * L
        self.enc_slot_p_ema  = torch.zeros(L)
        self.enc_slot_m_ema  = torch.zeros(L)
        self.enc_slot_m2_ema = torch.zeros(L)
        self.enc_slot_r_ema  = torch.zeros(L, 2)
        self.enc_slot_r2_ema = torch.zeros(L, 2)
        self.enc_slot_patience = torch.zeros(L, dtype=torch.long)
        self.enc_total_updates = 0  # 计数编码器额外优化步
        
        # 冻结时缓存的槽位参数（用于“被拿走”反演）
        self.slot_params_m = torch.zeros(L)       # m̂_i
        self.slot_params_r = torch.zeros(L, 2)    # r̂_i = [x,y]
        self._unlock_patience = torch.zeros(L, dtype=torch.long)

        # 全局快照（稳定前一状态）
        self._stable_M = None     # 标量
        self._stable_C = None     # 3维
        self._count_ema = 0.0     # g_pred 的 count EMA
        
        # 全局 g_pred 的 EMA（仅用模型输出）
        self.g_mass_ema = torch.tensor(0.0)   # 标量
        self.g_com_ema  = torch.zeros(2)      # [2]
        self.g_count_ema= torch.tensor(0.0)   # 标量

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        obs_history_shape,
        commands_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_history_shape,
            commands_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, obs_history, commands, critic_obs):
        critic_obs = torch.cat((critic_obs, commands), dim=-1)
        # act
        encoder_out = self.encoder.encode(obs_history)
        self.transition.actions = self.actor_critic.act(
            torch.cat((encoder_out, obs, commands), dim=-1)
        ).detach()

        # evaluate
        if self.critic_take_latent:
            critic_obs = torch.cat((critic_obs, encoder_out), dim=-1)
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()

        # storage
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_obs = critic_obs
        self.transition.observation_history = obs_history
        self.transition.commands = commands
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.transition.next_observations = next_obs
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        #初始化与数据准备
        num_updates = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl = 0
        generator = self.storage.mini_batch_generator(
            self.num_group,
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for (
            obs_batch,
            critic_obs_batch,
            obs_history_batch, _,
            group_commands_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            encoder_out_batch = self.encoder.encode(obs_history_batch)
            # 历史观测编码
            commands_batch = group_commands_batch
            # 控制命令
            # 策略网络前向传播
            self.actor_critic.act(
                torch.cat(
                    (encoder_out_batch, obs_batch, commands_batch),
                    dim=-1,
                )
            )
            # 获取当前策略的动作概率
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            # 价值网络评估
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            #KL散度计算（策略变化度量）
            kl_mean = torch.tensor(0, device=self.device, requires_grad=False)
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

            # 自适应学习率调整
            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            if self.desired_kl != None and self.early_stop:
                if kl_mean > self.desired_kl * 1.5:
                    print("early stop, num_updates =", num_updates)
                    break

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            # print(ratio)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss价值损失
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            entropy_batch_mean = entropy_batch.mean()
            #总损失函数代理损失（策略优化）
            # 价值损失（价值函数优化）
            # 熵奖励（鼓励探索）
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch_mean
            )
            # 学习率退火
            if self.anneal_lr:
                frac = 1.0 - num_updates / (
                    self.num_learning_epochs * self.num_mini_batches
                )
                self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

            # Gradient step梯度更新梯度裁剪​
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            num_updates += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl += kl_mean.item()

        num_updates_extra = 0
        mean_extra_loss = 0
        #编码器辅助训练
        if self.extra_optimizer is not None:
            generator = self.storage.encoder_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
            for (
                next_obs_batch,
                critic_obs_batch,
                obs_history_batch,
            ) in generator:
                if self.encoder.is_mlp_encoder:
                    #1）前向
                    
                    self.encoder.encode(obs_history_batch)
                    encode_batch = self.encoder.get_encoder_out()
                    g_pred, s_pred = self.encoder.split_outputs(encode_batch)   # g:[B,11], s:[B,L,4]
                    #槽位预测解耦​
                    exist_logit = s_pred[..., 0]
                    mass_pred   = F.softplus(s_pred[..., 1])
                    pos_pred    = s_pred[..., 2:4]
                    
                    
                    # 2) 目标(来自 critic_obs 的特权段)，仅作监督，禁止梯度回传
                    B = critic_obs_batch.size(0)
                    L = self.encoder.num_slots
                    #gt means ground truth
                    # critic 前10维 + count(紧随其后的priv首维)
                    g_tgt = torch.cat([critic_obs_batch[:, :10], critic_obs_batch[:, 10:11]], dim=-1).detach()  # [B,11]
                    #critic_obs_batch[:, 10:11]明确表示“取第 10 列且保留二维形状”，强调该列是​​独立特征​​（如物体数量），而非普通标量。
                    priv = critic_obs_batch[:, 10:10 + (1 + 4*L)]     # [B,1+4L]
                    slots_gt = priv[:, 1:].view(B, L, 4).detach()              # [B,L,4] -> [mass, rx, ry, rz]
                    #只要槽位的​​任一属性非零​​（质量、位置等至少一个不为零），即视为有效槽位（active=1）
                    active = (slots_gt.abs().sum(dim=-1) > 0)         # [B,L]有效槽位掩码 [B, L]
                    mass_gt = slots_gt[..., 0]
                    pos_gt  = slots_gt[..., 1:3]
                    #todo 忘记把encoder和actor_critic的device对齐了 就是说，是不是一个是概率，一个是三个维度的坐标
                    
                    
                    #3） losses
                    loss_g = (g_pred - g_tgt).pow(2).mean()# 全局预测MSE损失
                    bce = F.binary_cross_entropy_with_logits(exist_logit, active.float(), reduction='none').mean()# 存在性BCE损失
                    mass_mse = (mass_pred - mass_gt).pow(2) # 质量预测MSE损失
                    pos_mse  = (pos_pred  - pos_gt ).pow(2).sum(dim=-1) # 位置预测MSE损失
                    mask = active.float() # 仅对有效槽位计算损失
                    # 平均时除以有效槽位数，避免稀疏目标导致
                    loss_mass = (mass_mse * mask).sum() / (mask.sum() + 1e-8)
                    loss_pos  = (pos_mse  * mask).sum() / (mask.sum() + 1e-8)

                    extra_loss = loss_g + bce + loss_mass + loss_pos
                    # print(
                    #     f"[ENC] loss_g={loss_g.item():.6f} "
                    #     f"bce={bce.item():.6f} "
                    #     f"loss_mass={loss_mass.item():.6f} "
                    #     f"loss_pos={loss_pos.item():.6f} "
                    #     f"sum={extra_loss.item():.6f}"
                    # )
                    
                    
                    # # === 自适应冻结统计（按槽位）===
                    # with torch.no_grad():
                    #     # 每槽位的平均（只在active样本上）
                    #     per_slot_den = mask.sum(dim=0).clamp_min(1.0)        # [L]den是 denominator（分母）的缩写​​
                    #     #沿批次维度（dim=0）对每个槽位（L）的 mask值求和，得到​​每个槽位的有效样本数​​（形状 [L]）。
                    #     #例如：若批次大小 B=32，某槽位在16个批次中有效，则其值为16。
                    #     per_slot_mass = (mass_mse * mask).sum(dim=0) / per_slot_den  # [L] 槽位质量MSE均值
                    #     per_slot_pos  = (pos_mse  * mask).sum(dim=0) / per_slot_den  # [L] 槽位位置MSE均值
                    #     per_slot_p    = torch.sigmoid(exist_logit).mean(dim=0)  # [L]槽位平均存在概率

                    #     beta = self.enc_freeze_cfg["ema_beta"]      # EMA衰减因子（如0.9）
                    #     self.enc_slot_loss_mass_ema = beta * self.enc_slot_loss_mass_ema + (1 - beta) * per_slot_mass.cpu()
                    #     self.enc_slot_loss_pos_ema  = beta * self.enc_slot_loss_pos_ema  + (1 - beta) * per_slot_pos.cpu()
                    #     self.enc_slot_p_ema         = beta * self.enc_slot_p_ema         + (1 - beta) * per_slot_p.cpu()

                    #     # 判据
                    #     p_ok   = self.enc_slot_p_ema >= self.enc_freeze_cfg["p_thresh"]
                    #     mass_ok= self.enc_slot_loss_mass_ema <= self.enc_freeze_cfg["mass_mse_thresh"]
                    #     pos_ok = self.enc_slot_loss_pos_ema  <= self.enc_freeze_cfg["pos_mse_thresh"]
                    #     ok = (p_ok & mass_ok & pos_ok)

                    #     # 耐心计数
                    #     self.enc_slot_patience[ok] += 1
                    #     self.enc_slot_patience[~ok] = 0

                    #     # 触发冻结
                    #     can_freeze = self.enc_total_updates >= self.enc_freeze_cfg["min_updates"]
                    #     for i in range(L):
                    #         if not self.enc_slot_frozen[i] and can_freeze and self.enc_slot_patience[i] >= self.enc_freeze_cfg["patience"]:
                    #             self.encoder.freeze_slot_head(i, True)
                    #             self.enc_slot_frozen[i] = True
                    #             # 可选：将该槽位损失权重置零（不再计入 extra_loss）
                    #             # 也可打印日志到 stdout 或 TensorBoard
                    #             # print(f"[Encoder] Freeze slot head {i} at update {self.enc_total_updates}")
                    # === 自适应冻结统计（仅用模型输出，不依赖特权）===
                    with torch.no_grad():
   
                        # 预测信号
                        p = torch.sigmoid(exist_logit)        # [B,L]
                        m = mass_pred                         # [B,L]
                        r = pos_pred                          # [B,L,2]
                        # EMA 更新
                        beta = self.enc_freeze_cfg["ema_beta"]
                        # batch 平均（跨 B）再做 EMA（时间）
                        p_mean = p.mean(dim=0).cpu()               # [L] 存在概率均值
                        m_mean = m.mean(dim=0).cpu()               # [L] 质量均值
                        m2_mean= (m**2).mean(dim=0).cpu()          # [L] 质量平方均值
                        r_mean = r.mean(dim=0).cpu()               # [L,2] 位置均值
                        r2_mean= (r**2).mean(dim=0).cpu()          # [L,2] 位置平方均值

                        self.enc_slot_p_ema  = beta*self.enc_slot_p_ema  + (1-beta)*p_mean
                        self.enc_slot_m_ema  = beta*self.enc_slot_m_ema  + (1-beta)*m_mean
                        self.enc_slot_m2_ema = beta*self.enc_slot_m2_ema + (1-beta)*m2_mean
                        self.enc_slot_r_ema  = beta*self.enc_slot_r_ema  + (1-beta)*r_mean
                        self.enc_slot_r2_ema = beta*self.enc_slot_r2_ema + (1-beta)*r2_mean

                        # 时间方差估计（EMA 近似）
                        m_var = (self.enc_slot_m2_ema - self.enc_slot_m_ema**2).clamp_min(0.0)      # [L]
                        r_var = (self.enc_slot_r2_ema - self.enc_slot_r_ema**2).sum(dim=-1).clamp_min(0.0)  # [L]

                        # 判据（使用 enc_freeze_cfg 的阈值，但不再依赖特权 MSE）冻结条件
                        p_ok   = self.enc_slot_p_ema >= self.enc_freeze_cfg.get("p_thresh", 0.9)  # 槽存在概率阈值
                        mass_ok= m_var <= self.enc_freeze_cfg.get("mass_var_thresh", 1e-3)  # 质量时间方差阈值
                        pos_ok = r_var <= self.enc_freeze_cfg.get("pos_var_thresh", 1e-4)  # 位置时间方差阈值
                        ok = (p_ok & mass_ok & pos_ok)

                        # 耐心与冻结
                        self.enc_slot_patience[ok] += 1
                        self.enc_slot_patience[~ok] = 0
                        can_freeze = self.enc_total_updates >= self.enc_freeze_cfg.get("min_updates", 50)
                        patience_need = self.enc_freeze_cfg.get("patience", 5)
                        for i in range(self.encoder.num_slots):
                            if not self.enc_slot_frozen[i] and can_freeze and self.enc_slot_patience[i] >= patience_need:
                                self.encoder.freeze_slot_head(i, True)
                                self.enc_slot_frozen[i] = True
                                # 记录冻结时刻的槽位快照
                                self.slot_params_m[i] = float(self.enc_slot_m_ema[i].item())
                                self.slot_params_r[i] = self.enc_slot_r_ema[i].clone()
                        try:
                            g_mass_idx = int(self.enc_freeze_cfg.get("g_mass_idx", 0))
                            g_com_start= int(self.enc_freeze_cfg.get("g_com_start", 1))
                            g_com_dim  = int(self.enc_freeze_cfg.get("g_com_dim", 2))
                            if g_pred.shape[-1] >= max(g_mass_idx+1, g_com_start+g_com_dim):
                                g_mass = g_pred[:, g_mass_idx].mean().detach().cpu()
                                g_com  = g_pred[:, g_com_start:g_com_start+g_com_dim].mean(dim=0).detach().cpu()
                                self.g_mass_ema = beta*self.g_mass_ema + (1-beta)*g_mass
                                # 只取前2维为质心
                                g_com2 = g_com[:2] if g_com.numel() >= 2 else torch.zeros(2)
                                self.g_com_ema  = beta*self.g_com_ema  + (1-beta)*g_com2
                            # count 默认取最后一维
                            g_count = g_pred[:, -1].mean().detach().cpu()
                            self.g_count_ema = beta*self.g_count_ema + (1-beta)*g_count
                        except Exception:
                            pass

                        # === 解冻判据 1：p 过低（被拿走）===
                        unlock_p_thresh = self.enc_freeze_cfg.get("unlock_p_thresh", 0.3)
                        unlock_patience = int(self.enc_freeze_cfg.get("unlock_patience", 10))
                        for i in range(self.encoder.num_slots):
                            if self.enc_slot_frozen[i]:
                                if self.enc_slot_p_ema[i] < unlock_p_thresh:
                                    self._unlock_patience[i] += 1
                                else:
                                    self._unlock_patience[i] = 0
                                if self._unlock_patience[i] >= unlock_patience:
                                    self.encoder.freeze_slot_head(i, False)  # 解冻
                                    self.enc_slot_frozen[i] = False
                                    self._unlock_patience[i] = 0
                                    # 解冻后清空该槽的稳定计数
                                    self.enc_slot_patience[i] = 0

                        # === 解冻判据 2：全局一致性（质量/质心）===
                        if self.enc_freeze_cfg.get("unlock_by_consistency", True):
                            # 仅当存在已冻结槽位时才考虑
                            if any(self.enc_slot_frozen):
                                # 槽位加权和
                                ci = (self.enc_slot_m_ema * self.enc_slot_p_ema)        # [L]
                                sum_m = float(ci.sum().item()) + 1e-8
                                sum_mr = (ci.unsqueeze(-1) * self.enc_slot_r_ema).sum(dim=0)  # [2]
                                com_slots = sum_mr / sum_m

                                Mg = float(self.g_mass_ema.item())
                                Cg = self.g_com_ema

                                # 容差
                                mass_margin = max(self.enc_freeze_cfg.get("mass_margin_frac", 0.1)*max(Mg, 1e-6), 1e-6)
                                com_margin  = float(self.enc_freeze_cfg.get("com_margin", 0.05))

                                mass_bad = (sum_m - Mg) > mass_margin  # 槽和过大（可能减少了负载但槽位仍冻结）
                                com_bad  = torch.norm((com_slots - Cg).float(), p=2).item() > com_margin

                                if (mass_bad or com_bad) and (self.enc_total_updates >= self.enc_freeze_cfg.get("min_updates", 50)):
                                    # 选一个冻结槽位，使去掉它后与全局更一致
                                    alpha = float(self.enc_freeze_cfg.get("alpha_mass", 1.0))
                                    betaC = float(self.enc_freeze_cfg.get("beta_com", 1.0))
                                    best_i = None
                                    best_cost = float("inf")

                                    for i in range(self.encoder.num_slots):
                                        if not self.enc_slot_frozen[i]:
                                            continue
                                        ci_i = float(ci[i].item())
                                        # 假设“拿走”该槽
                                        M_wo = max(sum_m - ci_i, 1e-8)
                                        C_wo = (sum_mr - ci_i * self.enc_slot_r_ema[i]) / M_wo
                                        cost = alpha * abs(M_wo - Mg) + betaC * torch.norm((C_wo - Cg).float(), p=2).item()
                                        if cost < best_cost:
                                            best_cost = cost
                                            best_i = i
                                    if best_i is not None:
                                        # 解冻候选槽位（需要耐心避免抖动）
                                        self._unlock_patience[best_i] += 1
                                        # 其他候选清零
                                        for j in range(self.encoder.num_slots):
                                            if j != best_i:
                                                self._unlock_patience[j] = 0
                                        if self._unlock_patience[best_i] >= unlock_patience:
                                            self.encoder.freeze_slot_head(best_i, False)
                                            self.enc_slot_frozen[best_i] = False
                                            self._unlock_patience[best_i] = 0
                                            self.enc_slot_patience[best_i] = 0
                        
                    
                    
                    
                    
                    
                    
                    
                else:
                    extra_loss = torch.zeros((), device=self.device)
                    # 统一到下方优化步骤
                    self.extra_optimizer.zero_grad()
                    extra_loss.backward()
                    nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                    self.extra_optimizer.step()
                    num_updates_extra += 1
                    mean_extra_loss += extra_loss.item()
                    continue  # 非 MLP 编码器时直接继续
                    

                # 5) 反传与优化
                self.extra_optimizer.zero_grad()
                extra_loss.backward()
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                self.extra_optimizer.step()

                num_updates_extra += 1
                mean_extra_loss += extra_loss.item()
                self.enc_total_updates = getattr(self, "enc_total_updates", 0) + 1
        #性能统计与清理
        mean_value_loss /= num_updates
        if num_updates_extra > 0:
            mean_extra_loss /= num_updates_extra
        mean_surrogate_loss /= num_updates
        mean_kl /= num_updates
        self.storage.clear()

        return (mean_value_loss, mean_extra_loss, mean_surrogate_loss, mean_kl)


    def get_frozen_slot_count(self) -> int:
        """返回已冻结的槽位头数量"""
        return int(sum(1 for f in self.enc_slot_frozen if f))