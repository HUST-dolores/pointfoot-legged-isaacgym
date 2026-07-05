function fig4_scenario_vs_motor()
% Fig4 — 电机的 "no free torque" (定腿 ξ*=(0.914,0.8) 扫电机: 微调 vs 零样本)。
% A: 综合 fitness 有内部最优 k≈1.0 (=3D BO 的 ξ*电机), 仅微调后显现(零样本饱和)。
% B: 机制 = 质量型(obstacle/accel)随电机变重而↓ · 扭矩型(slope/load)↑ → 交汇出内部最优。
% 电机质量是物理的(ENCOS motor_mass_delta_kg 加到关节 link)。
D = cd_load();
mk = D.msweep_motor(:); mfit = D.msweep_fit(:);      % 零样本 11 点
fk = D.ftmotor_motor(:); ffit = D.ftmotor_fit(:);    % 微调 6 点
fscen = D.ftmotor_scen;                              % 6×4
SCOL = [0.106 0.62 0.467; 0.851 0.373 0.008; 0.459 0.439 0.702; 0.906 0.161 0.541];
names = {'越障 obstacle', '爬坡 slope', '负载 load', '加速 accel'};
f = figure('Position', [80 80 1200 450], 'Color', 'w');
% A: 综合 fitness
ax = subplot(1, 2, 1); hold(ax, 'on'); box(ax, 'on');
plot(ax, fk, ffit, '-o', 'Color', [0.70 0.09 0.17], 'LineWidth', 2.2, ...
    'MarkerFaceColor', [0.70 0.09 0.17], 'MarkerSize', 8);
plot(ax, mk, mfit, '-s', 'Color', [0.26 0.58 0.76], 'LineWidth', 1.8, ...
    'MarkerFaceColor', [0.26 0.58 0.76], 'MarkerSize', 5);
[~, bi] = max(ffit);
xline(fk(bi), ':', 'Color', [0.70 0.09 0.17], 'HandleVisibility', 'off');
text(ax, fk(bi) + 0.05, ffit(bi) - 8, sprintf('内部最优 k≈%.1f (=ξ*电机)', fk(bi)), ...
    'Color', [0.70 0.09 0.17], 'FontSize', 9);
xlabel(ax, '电机扭矩缩放 k (+ ENCOS 物理质量代价)');
ylabel(ax, '综合 fitness (4场景等权, 定腿 ξ*=(0.914,0.8))');
title(ax, '综合 fitness 有内部最优 (仅微调后显现)', 'FontSize', 10);
legend(ax, {'微调 (为每个设计 +400步)', '零样本 (固定 B'' generalist)'}, ...
    'Location', 'southeast', 'FontSize', 9); grid(ax, 'on');
% B: 逐场景机制 (相对 k=0.6 的 % 变化)
ax2 = subplot(1, 2, 2); hold(ax2, 'on'); box(ax2, 'on');
rel = (fscen ./ fscen(1, :) - 1) * 100;
styles = {'--o', '-o', '-o', '--o'};   % 质量型虚线(obstacle/accel), 扭矩型实线(slope/load)
h = gobjects(1, 4);
for k = 1:4
    h(k) = plot(ax2, fk, rel(:, k), styles{k}, 'Color', SCOL(k, :), 'LineWidth', 1.8, ...
        'MarkerFaceColor', SCOL(k, :), 'MarkerSize', 6);
end
yline(0, 'k', 'HandleVisibility', 'off');
xlabel(ax2, '电机扭矩缩放 k'); ylabel(ax2, '逐场景 相对 k=0.6 的变化 (%)');
title(ax2, '机制: 质量型 obstacle/accel↓ · 扭矩型 slope/load↑', 'FontSize', 10);
legend(ax2, h, names, 'Location', 'east', 'NumColumns', 2, 'FontSize', 8); grid(ax2, 'on');
sgtitle('电机的 "no free torque": 弱电机差、最大电机不更好、最优 k≈1.0-1.1 (质量代价 vs 扭矩收益 的交汇)');
cd_export(f, 'fig4_scenario_vs_motor');
end
