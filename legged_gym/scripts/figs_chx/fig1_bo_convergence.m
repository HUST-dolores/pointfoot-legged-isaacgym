function fig1_bo_convergence()
% Fig1 — BO 收敛曲线: 2D(仅腿长) vs 3D(腿长+电机)
% 每候选 = 从预训练 generalist 微调400步 + 单形态4场景eval; fitness=4场景等权 mean_rew。
D = cd_load();
f = figure('Position', [100 100 1100 420], 'Color', 'w');
dims = {'2d', '3d'};
titles = {'2D BO (仅腿长)', '3D BO (腿长 + 电机)'};
cols = {[0.13 0.40 0.67], [0.70 0.09 0.17]};
for i = 1:2
    ax = subplot(1, 2, i); hold(ax, 'on'); box(ax, 'on');
    fit = D.(['bo' dims{i} '_fit']);
    order = D.(['bo' dims{i} '_order']);
    isinit = logical(D.(['bo' dims{i} '_isinit']));
    [~, o] = sort(order); fit = fit(o); isinit = isinit(o);
    x = 1:numel(fit);
    best = cummax(fit);
    stairs(ax, x, best, 'Color', cols{i}, 'LineWidth', 2);
    scatter(ax, x(isinit), fit(isinit), 55, [.53 .53 .53], 'filled', 'MarkerEdgeColor', 'k');
    scatter(ax, x(~isinit), fit(~isinit), 62, cols{i}, 'filled', 'MarkerEdgeColor', 'k');
    xline(sum(isinit) + 0.5, '--', 'Color', [.5 .5 .5], 'HandleVisibility', 'off');
    [bv, bi] = max(fit);
    text(ax, x(bi), bv, sprintf('  \\xi^* fit=%.1f', bv), 'FontWeight', 'bold', 'FontSize', 10);
    title(ax, titles{i}); xlabel(ax, '评估序号 (每候选微调400步)');
    ylabel(ax, 'fitness  f(\xi) = 4场景等权 mean\_rew'); grid(ax, 'on');
    legend(ax, {'best-so-far', '初始采样', 'BO提议(EI)'}, 'Location', 'southeast', 'FontSize', 8);
end
sgtitle('BO 收敛曲线 — 预训练 generalist 微调400步/候选');
cd_export(f, 'fig1_bo_convergence');
end
