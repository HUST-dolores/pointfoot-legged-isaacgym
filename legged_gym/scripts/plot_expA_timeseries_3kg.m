function plot_expA_timeseries_3kg(variant)
% 实验A — Figure：在 3kg 力 / 3kg 真实质量下的估计时间序列（横轴=时间）。
% 上=真实负载 3kg，下=竖直向下力 3kg 当量（无真实负载）；黑色阶跃为参考真值。
% 默认变体 Model-guided，可传入其它变体名。

if nargin < 1 || isempty(variant); variant = 'Model-guided'; end
[R, EXPORT_DIR] = expA_load_runs();
cMass = [0 0.45 0.74]; cForce = [0.85 0.33 0.10];

rms = expA_pick(R, variant, 'walk', 'mass_single');
rfs = expA_pick(R, variant, 'walk', 'force_single');

fig = figure('Color', 'w', 'Position', [80 80 920 620], 'Name', 'Exp A 3kg 时间序列');

ax1 = subplot(2, 1, 1); hold(ax1, 'on'); grid(ax1, 'on'); box(ax1, 'on');
if ~isempty(rms)
    stairs(ax1, rms.t, rms.ref_ts, 'k-', 'LineWidth', 1.4);
    plot(ax1, rms.t, rms.rl_ts, '-', 'Color', cMass, 'LineWidth', 1.4);
    plot(ax1, rms.t, rms.qs_ts, ':', 'Color', [.5 .5 .5], 'LineWidth', 1.0);
    legend(ax1, {'真实质量 (3kg 阶跃)', 'RL 编码器', 'QS'}, 'Location', 'northeast', 'FontSize', 8);
else
    text(ax1, 0.5, 0.5, sprintf('缺少 %s 的 mass\\_single(3kg) run', variant), 'Units', 'normalized', 'HorizontalAlignment', 'center');
end
title(ax1, sprintf('%s — 真实负载 3kg', variant)); ylabel(ax1, 'mass [kg]'); ylim(ax1, [-2 10]);

ax2 = subplot(2, 1, 2); hold(ax2, 'on'); grid(ax2, 'on'); box(ax2, 'on');
if ~isempty(rfs)
    stairs(ax2, rfs.t, rfs.ref_ts, 'k-', 'LineWidth', 1.4);
    plot(ax2, rfs.t, rfs.rl_ts, '-', 'Color', cForce, 'LineWidth', 1.4);
    plot(ax2, rfs.t, rfs.qs_ts, ':', 'Color', [.5 .5 .5], 'LineWidth', 1.0);
    legend(ax2, {'力当量 (3kg 阶跃, 无负载)', 'RL 编码器', 'QS'}, 'Location', 'northeast', 'FontSize', 8);
else
    text(ax2, 0.5, 0.5, sprintf('缺少 %s 的 force\\_single(3kg) run', variant), 'Units', 'normalized', 'HorizontalAlignment', 'center');
end
title(ax2, sprintf('%s — 竖直向下力 3kg 当量（无负载）', variant));
xlabel(ax2, 'time [s]'); ylabel(ax2, 'mass [kg]'); ylim(ax2, [-2 10]);

sgtitle('实验A：3kg 力 / 质量下的估计时间序列 —— 力被读成幻象质量', 'FontWeight', 'bold');
expA_savefig(fig, fullfile(EXPORT_DIR, sprintf('expA_timeseries_3kg_%s', strrep(variant, '-', '_'))));
end
