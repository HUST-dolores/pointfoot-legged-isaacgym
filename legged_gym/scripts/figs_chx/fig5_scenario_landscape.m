function fig5_scenario_landscape()
% Fig5 — 逐场景性能地形 vs 腿长 (2D BO 12点, 电机固定=1.0, scatteredInterpolant 线性插值)。
% 各场景均偏短腿(低CoM稳); 负载/爬坡对腿长最敏感, 越障相对不敏感(固定净高proxy无"够得着"激励)。
D = cd_load();
th = D.bo2d_thigh(:); sh = D.bo2d_shank(:); scen = D.bo2d_scen; fit = D.bo2d_fit(:);
[~, bi] = max(fit);
names = {'越障 obstacle', '爬坡 slope', '负载 load', '加速 accel', '综合 fitness'};
Z = {scen(:, 1), scen(:, 2), scen(:, 3), scen(:, 4), fit};
[gx, gy] = meshgrid(linspace(0.8, 1.2, 60), linspace(0.8, 1.2, 60));
f = figure('Position', [40 80 1900 380], 'Color', 'w');
for p = 1:5
    ax = subplot(1, 5, p); hold(ax, 'on'); box(ax, 'on');
    F = scatteredInterpolant(th, sh, Z{p}, 'linear', 'none');
    gz = F(gx, gy);
    contourf(ax, gx, gy, gz, 12, 'LineStyle', 'none');
    scatter(ax, th, sh, 16, 'w', 'filled', 'MarkerEdgeColor', 'k');
    scatter(ax, th(bi), sh(bi), 200, 'r', 'p', 'filled', 'MarkerEdgeColor', 'k');
    colorbar(ax); colormap(ax, parula);
    title(ax, names{p}); xlabel(ax, '大腿缩放 s\_t');
    if p == 1, ylabel(ax, '小腿缩放 s\_s'); end
    xlim(ax, [0.8 1.2]); ylim(ax, [0.8 1.2]); axis(ax, 'square');
end
sgtitle('逐场景性能地形 vs 腿长 (2D BO) — 各场景均偏短腿; 负载/爬坡最敏感, 越障不敏感 \rightarrow \xi^*落短腿(红星)');
cd_export(f, 'fig5_scenario_landscape');
end
