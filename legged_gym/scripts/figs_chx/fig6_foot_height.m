function fig6_foot_height()
% Fig6 [路B-A] — 抬腿峰值净高 vs 腿长 (全 obstacle 场景, B' 零样本, 按形态聚合+三角插值)。
% 长腿抬得更高但更易摔 -> 固定净高 proxy 漏掉的 "够得着 vs 稳" 权衡。
D = cd_load();
tu = D.footh_bth(:); su = D.footh_bsh(:);
peak = D.footh_bpeak(:); fall = D.footh_bfall(:);
target = D.footh_target;
[gx, gy] = meshgrid(linspace(0.8, 1.2, 60), linspace(0.8, 1.2, 60));
f = figure('Position', [80 80 1200 460], 'Color', 'w');
Z = {peak, fall};
ttl = {sprintf('峰值抬脚净高 (m) — 命令目标 %.2fm', target), '摔倒率 (%)'};
cbl = {'峰值净高 (m)', 'fall %'};
cmp = {'parula', 'hot'};
for p = 1:2
    ax = subplot(1, 2, p); hold(ax, 'on'); box(ax, 'on');
    F = scatteredInterpolant(tu, su, Z{p}, 'linear', 'none');
    contourf(ax, gx, gy, F(gx, gy), 14, 'LineStyle', 'none');
    scatter(ax, tu, su, 6, 'w', 'filled', 'MarkerEdgeColor', 'k');
    c = colorbar(ax); c.Label.String = cbl{p};
    colormap(ax, cmp{p});
    title(ax, ttl{p}); xlabel(ax, '大腿缩放 s\_t');
    if p == 1, ylabel(ax, '小腿缩放 s\_s'); end
    xlim(ax, [0.8 1.2]); ylim(ax, [0.8 1.2]); axis(ax, 'square');
end
sgtitle('抬腿净高 vs 腿长 (全obstacle, generalist零样本) — 长腿抬得更高但更易摔 \rightarrow proxy漏掉的 够得着-vs-稳 权衡');
cd_export(f, 'fig6_foot_height');
end
