function expA_transfer_curves()
% 实验A — Figure：各变体“真实质量 / 向下力”转移曲线（RL 编码器估计 vs 参考真值）。
% 每个变体一个子图：散点画全 1–6kg（展示分布外饱和），黄色阴影=训练分布区间 [2,4]kg，
% 拟合线只用 [2,4] 分布内的点（与 expA_load_runs 的“每段去前 1s 响应”一致）。
% 蓝=真实质量、红=向下力；两线越接近越分不清。per-env：每个环境一个散点，不跨环境平均。

BAND = [2 4];                       % 训练分布区间（拟合 & 阴影）
[R, EXPORT_DIR] = expA_load_runs(); % 默认每段去前 1.0s 响应
variants = {'Model-guided', 'Source-guided', 'Estimate-guided', 'RL-only'};
cMass = [0 0.45 0.74]; cForce = [0.85 0.33 0.10]; xl = [0 7]; yl = [-1 8];

fig = figure('Color', 'w', 'Position', [60 60 1000 780], 'Name', 'Exp A 转移曲线');
for vi = 1:4
    ax = subplot(2, 2, vi); hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    % 分布内阴影 [2,4]
    patch(ax, [BAND(1) BAND(2) BAND(2) BAND(1)], [yl(1) yl(1) yl(2) yl(2)], ...
        [0.95 0.9 0.55], 'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(ax, xl, xl, '--', 'Color', [.6 .6 .6], 'HandleVisibility', 'off');  % y=x 参考
    h = gobjects(0); lab = {};
    rm = expA_pick(R, variants{vi}, 'walk', 'mass_sweep');
    rf = expA_pick(R, variants{vi}, 'walk', 'force_sweep');
    if ~isempty(rm)
        scatter(ax, rm.ref, rm.rl, 26, 'o', 'MarkerEdgeColor', cMass, 'HandleVisibility', 'off');
        [a, b] = fit_band(rm.ref, rm.rl, BAND);
        h(end+1) = plot(ax, BAND, a*BAND+b, '-', 'Color', cMass, 'LineWidth', 2.2); %#ok<AGROW>
        lab{end+1} = sprintf('真实质量[2,4]: %.2f·x %+.2f', a, b); %#ok<AGROW>
    end
    if ~isempty(rf)
        scatter(ax, rf.ref, rf.rl, 26, 'd', 'MarkerEdgeColor', cForce, 'HandleVisibility', 'off');
        [a, b] = fit_band(rf.ref, rf.rl, BAND);
        h(end+1) = plot(ax, BAND, a*BAND+b, '-', 'Color', cForce, 'LineWidth', 2.2); %#ok<AGROW>
        lab{end+1} = sprintf('向下力[2,4]: %.2f·x %+.2f', a, b); %#ok<AGROW>
    end
    axis(ax, [xl yl]);
    title(ax, variants{vi}, 'FontWeight', 'bold');
    xlabel(ax, '参考真值 [kg]（真实质量 / 力当量）'); ylabel(ax, 'RL 编码器估计 [kg]');
    if ~isempty(h); legend(ax, h, lab, 'Location', 'northwest', 'FontSize', 8); end
end
sgtitle('实验A：RL 编码器转移曲线（行走，每段去前1s）—— 阴影=训练分布[2,4]，拟合用分布内点；两线越接近越分不清', 'FontWeight', 'bold');
expA_savefig(fig, fullfile(EXPORT_DIR, 'expA_transfer_curves'));
end

% ------------------------------------------------------------------
function [a, b] = fit_band(x, y, band)
x = x(:); y = y(:);
sel = x >= band(1) & x <= band(2) & isfinite(x) & isfinite(y);
[a, b] = expA_linfit(x(sel), y(sel));
end
