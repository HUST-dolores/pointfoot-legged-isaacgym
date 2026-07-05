function fig3_finetune_curve()
% Fig3 — 微调效率: generalist 从 B 微调 N 步 逼近 specialist(从头专训该形态 +6000步)。
% 单形态 xi=(1.2,0.8) 最极端角落, 4场景固定难度0.3, fitness=4场景等权。
D = cd_load();
steps = D.ft_steps; fit = D.ft_fit; scen = D.ft_scen;   % scen: 7x4
[steps, o] = sort(steps); fit = fit(o); scen = scen(o, :);
isspec = steps >= 6000;
fts = steps(~isspec); ftf = fit(~isspec);
spec = fit(find(isspec, 1));
SCOL = [0.106 0.62 0.467; 0.851 0.373 0.008; 0.459 0.439 0.702; 0.906 0.161 0.541];
names = {'越障 obstacle', '爬坡 slope', '负载 load', '加速 accel'};
f = figure('Position', [80 80 1150 430], 'Color', 'w');
% 左: 综合 fitness
ax = subplot(1, 2, 1); hold(ax, 'on'); box(ax, 'on');
yline(spec, '--', sprintf('specialist上界(+6000)=%.1f', spec), 'Color', [.27 .27 .27], 'LineWidth', 1.5, 'HandleVisibility', 'off');
plot(ax, fts, ftf, '-o', 'Color', [0.70 0.09 0.17], 'LineWidth', 2, ...
    'MarkerFaceColor', [0.70 0.09 0.17], 'MarkerSize', 7);
for j = 1:numel(fts)
    text(ax, fts(j), ftf(j), sprintf('  %.0f%%', ftf(j)/spec*100), 'Color', [0.70 0.09 0.17], 'FontSize', 8);
end
xline(400, ':', 'Color', [.5 .5 .5], 'HandleVisibility', 'off');
xlabel(ax, '微调步数 (iter)'); ylabel(ax, 'fitness (单形态 \xi=(1.2,0.8), 难度0.3)');
title(ax, '微调效率: +400步=specialist 89%, +1500步追平'); grid(ax, 'on');
% 右: 逐场景
ax2 = subplot(1, 2, 2); hold(ax2, 'on'); box(ax2, 'on');
for k = 1:4
    plot(ax2, fts, scen(~isspec, k), '-o', 'Color', SCOL(k, :), 'LineWidth', 1.6, ...
        'MarkerFaceColor', SCOL(k, :), 'MarkerSize', 5);
    yline(scen(find(isspec, 1), k), ':', 'Color', SCOL(k, :), 'HandleVisibility', 'off');
end
xlabel(ax2, '微调步数 (iter)'); ylabel(ax2, '逐场景 mean\_rew');
title(ax2, '逐场景微调轨迹 (虚线=specialist)'); legend(ax2, names, 'Location', 'east', 'FontSize', 8); grid(ax2, 'on');
sgtitle('微调效率验证 — pretraining-finetuning 框架成立');
cd_export(f, 'fig3_finetune_curve');
end
