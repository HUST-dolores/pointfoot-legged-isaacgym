function fig2_table2_bars()
% Fig2 — Table-II 复现: spatial-DR(B) vs No-DR, 逐场景 + 综合。
% seed2 · 重标定 slope[1,10] (坡场景有区分度)。误差棒 = 100留出形态 bucket均值的 std。
D = cd_load();
f = figure('Position', [80 80 1200 470], 'Color', 'w');
tags = {'s2_d05', 's2_d03'};
titles = {'seed2 · 难度0.5 · 重标定slope[1,10]', 'seed2 · 难度0.3 · 重标定slope[1,10]'};
cats = {'越障 obstacle', '爬坡 slope', '负载 load', '加速 accel', '综合 fitness'};
for i = 1:2
    ax = subplot(1, 2, i); hold(ax, 'on'); box(ax, 'on');
    B = D.(['t2_B_' tags{i} '_mean']);   Bs = D.(['t2_B_' tags{i} '_std']);
    N = D.(['t2_noDR_' tags{i} '_mean']); Ns = D.(['t2_noDR_' tags{i} '_std']);
    Bv = [B mean(B)]; Nv = [N mean(N)]; Be = [Bs 0]; Ne = [Ns 0];
    x = 1:5; w = 0.38;
    bar(ax, x - w/2, Bv, w, 'FaceColor', [0.13 0.40 0.67], 'EdgeColor', 'k');
    bar(ax, x + w/2, Nv, w, 'FaceColor', [0.84 0.38 0.30], 'EdgeColor', 'k');
    errorbar(ax, x - w/2, Bv, Be, 'k', 'LineStyle', 'none', 'CapSize', 4);
    errorbar(ax, x + w/2, Nv, Ne, 'k', 'LineStyle', 'none', 'CapSize', 4);
    for j = 1:5
        d = Bv(j) - Nv(j);
        if d > 0, c = [0.09 0.51 0.09]; else, c = [0.70 0 0]; end
        text(ax, x(j), max(Bv(j), Nv(j)), sprintf('%+.0f', d), 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', 'Color', c, 'FontWeight', 'bold', 'FontSize', 9);
    end
    yline(0, 'k', 'HandleVisibility', 'off');
    set(ax, 'XTick', x, 'XTickLabel', cats); xtickangle(ax, 12);
    ylabel(ax, '100 留出形态 mean-of-bucket-means'); title(ax, titles{i}); grid(ax, 'on');
    if i == 1
        legend(ax, {'spatial-DR (B, 64随机腿形态)', 'No-DR (标称1.0/1.0)'}, 'Location', 'north', 'FontSize', 8.5);
    end
end
sgtitle('Table-II 复现: spatial-DR 泛化 击败 No-DR (逐场景 + 综合)');
cd_export(f, 'fig2_table2_bars');
end
