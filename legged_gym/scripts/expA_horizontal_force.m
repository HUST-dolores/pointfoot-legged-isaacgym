function expA_horizontal_force()
% 实验2 — Figure：力方向 × 变体 的"幻象质量斜率"对比（分布内 [2,4]，每段去前 1s）。
% 竖直 down 的斜率高(~0.7-0.8)=被读成质量；水平 fwd/left 斜率≈0=不被读成质量 → 方向特异性。
% 注：水平力本质是速度扰动，高幅值会把机器人推离指令(max|vx|~7-8 m/s)；Model-guided fwd
% 若偏高是被推动失稳的 artifact，非真混淆。

[R, EXPORT_DIR] = expA_load_runs();   % 默认每段去前 1s
variants = {'Model-guided', 'Source-guided', 'Estimate-guided', 'RL-only'};
dirs  = {'force_sweep', 'force_sweep_fwd', 'force_sweep_left'};
dlab  = {'竖直 down', '前向 fwd', '侧向 left'};
BAND = [2 4];

S = nan(numel(variants), numel(dirs));
for vi = 1:numel(variants)
    for di = 1:numel(dirs)
        r = expA_pick(R, variants{vi}, 'walk', dirs{di});
        if ~isempty(r)
            x = r.ref(:); y = r.rl(:);
            sel = x >= BAND(1) & x <= BAND(2) & isfinite(x) & isfinite(y);
            [a, ~] = expA_linfit(x(sel), y(sel));
            S(vi, di) = a;
        end
    end
end

fig = figure('Color', 'w', 'Position', [80 80 820 460], 'Name', 'Exp2 力方向特异性');
b = bar(S); grid on; box on;
set(gca, 'XTickLabel', variants, 'XTickLabelRotation', 15);
ylabel('幻象质量转移斜率（[2,4]）'); ylim([-0.2 1.0]);
yline(0, '-', 'Color', [.5 .5 .5], 'HandleVisibility', 'off');
legend(b, dlab, 'Location', 'northeast');
title({'实验2：力方向 vs 幻象质量斜率', ...
       '竖直力被读成质量(~0.7-0.8)，水平力不被读成质量(~0) → 方向特异性'}, 'FontWeight', 'bold');
expA_savefig(fig, fullfile(EXPORT_DIR, 'expA_horizontal_force_slope'));

% 控制台
fprintf('\n==== 力方向 × 变体 幻象质量斜率（行走, 分布内[2,4], 去前1s）====\n');
fprintf('%-16s | %8s %8s %8s\n', '变体', '竖直', '前向', '侧向');
for vi = 1:numel(variants)
    fprintf('%-16s | %8.2f %8.2f %8.2f\n', variants{vi}, S(vi, 1), S(vi, 2), S(vi, 3));
end
fprintf('提示：水平力高幅值会把机器人推离速度指令（max|vx|~7-8 m/s），Model-guided 前向偏高为失稳 artifact。\n');
end
