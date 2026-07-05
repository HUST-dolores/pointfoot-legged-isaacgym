function fig7_robustness()
% Fig7 — ξ* 鲁棒性: 报成"区域"而非过度精确的点。
% A: 最优是平坦盆地(前5名 fitness 差<3≈eval噪声, 却横跨 motor 0.89-1.46 → 电机维弱约束)。
% B: 权重敏感(改 fitness 权重, ξ* 只在小范围动: shank恒0.8, thigh 0.8-0.91, motor 0.96-1.1)。
% 稳健结论 = 短腿(shank≈0.8) + 中等电机(~1.0)。数据纯来自已有 3D BO 21个候选(无新训练)。
D = cd_load();
fit = D.fig7_fit(:); mo = D.fig7_mo(:);
schemes = D.fig7_schemes;              % 5x3 (thigh,shank,motor)
names = cellstr(D.fig7_scheme_names);
f = figure('Position', [80 80 1250 440], 'Color', 'w');
% A: 平坦盆地
ax = subplot(1, 2, 1); hold(ax, 'on'); box(ax, 'on');
[fs, order] = sort(fit, 'descend'); ms = mo(order);
x = (1:numel(fs))'; best = fs(1); band = fs >= best - 3;
bar(ax, x(band), fs(band), 'FaceColor', [0.70 0.09 0.17]);
bar(ax, x(~band), fs(~band), 'FaceColor', [0.73 0.73 0.73]);
yline(best - 3, '--', 'Color', [0.70 0.09 0.17], 'HandleVisibility', 'off');
idx = find(band);
for k = 1:numel(idx)
    i = idx(k);
    text(ax, x(i), fs(i) + 1, sprintf('k=%.2f', ms(i)), 'Rotation', 90, 'FontSize', 7, ...
        'Color', [0.48 0.04 0.04], 'HorizontalAlignment', 'left');
end
xlabel(ax, '候选排名 (fitness降序)'); ylabel(ax, '综合 fitness'); ylim(ax, [110 235]);
title(ax, sprintf('最优平坦盆地: 前%d名差<3, 横跨 motor 0.89-1.46', sum(band)), 'FontSize', 9.5);
legend(ax, {'最优盆地(差<3)', '其余候选'}, 'Location', 'northeast', 'FontSize', 8); grid(ax, 'on');
% B: 权重敏感
ax2 = subplot(1, 2, 2); hold(ax2, 'on'); box(ax2, 'on');
w = 0.25; xb = (1:size(schemes, 1));
cols = [0.106 0.62 0.467; 0.459 0.439 0.702; 0.902 0.16 0.29];
labs = {'大腿 thigh', '小腿 shank', '电机 motor'};
hh = gobjects(1, 3);
for j = 1:3
    hh(j) = bar(ax2, xb + (j - 2) * w, schemes(:, j), w, 'FaceColor', cols(j, :), 'EdgeColor', 'k');
end
yline(1.0, ':', 'Color', [.5 .5 .5], 'HandleVisibility', 'off');
set(ax2, 'XTick', xb, 'XTickLabel', names); xtickangle(ax2, 12);
ylabel(ax2, '最优设计值 (缩放)'); ylim(ax2, [0.7 1.25]);
title(ax2, '权重敏感: shank恒0.8, thigh 0.8-0.91, motor 0.96-1.1', 'FontSize', 9.5);
legend(ax2, hh, labs, 'Location', 'north', 'NumColumns', 3, 'FontSize', 8); grid(ax2, 'on');
sgtitle('ξ* 鲁棒性: 报成"区域"(短腿 shank≈0.8 + 中等电机~1.0) — 电机维弱约束(盆地平坦) + 权重稳健');
cd_export(f, 'fig7_robustness');
end
