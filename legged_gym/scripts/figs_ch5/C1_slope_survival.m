function C1_slope_survival()
% C1 斜坡存活率热图(四变体 × 坡度×负载),拆 2 张图(model-based 组 / 对照组)。
% 指标=末态直立率(末5s内≥80%时间 |pitch|<25°&|roll|<25° 的 env 占比)。
% 【数据已写死(2026-06-14)】baseline seed(Model/Estimate/RL=seed1,Source=seed42)的坡度×负载抗倾覆成功率热图;
%   口径同原版(walk_vx0.5, load2-30, 100-env battery);不再读 play 文件。配色/字体内联,零外部依赖。不自动保存。
variants = {'Model-guided', 'Estimate-guided', 'Source-guided', 'RL-only'};
loadEdges = [2 8 14 20 26 30.1]; loadCtr = (loadEdges(1:end-1)+loadEdges(2:end))/2;  % [5 11 17 23 28] kg
slopes = [8 12 16 20 22 24 26 28]; nv = numel(variants);
set(groot, 'defaultAxesFontName', 'Noto Sans CJK SC', 'defaultTextFontName', 'Noto Sans CJK SC');
% 热图 M(行=坡度[8 12 16 20 22 24 26 28]°, 列=负载箱中心 loadCtr),逐变体:
Mall = { ...
 [1.00 1.00 1.00 1.00 1.00; 1.00 1.00 1.00 1.00 1.00; 1.00 1.00 1.00 0.99 0.99; 1.00 1.00 1.00 0.95 1.00; 1.00 0.88 0.94 0.82 0.88; 0.71 0.79 0.58 0.63 0.92; 0.39 0.42 0.18 0.20 0.09; 0.14 0.10 0.10 0.00 0.00], ...  % Model-guided
 [1.00 1.00 1.00 1.00 1.00; 1.00 1.00 1.00 1.00 1.00; 1.00 1.00 1.00 1.00 1.00; 1.00 0.96 0.96 0.96 1.00; 1.00 0.75 0.88 0.83 0.81; 0.78 0.78 0.81 0.64 0.46; 0.48 0.36 0.27 0.36 0.56; 0.04 0.00 0.00 0.00 0.00], ...  % Estimate-guided
 [1.00 1.00 1.00 1.00 1.00; 1.00 1.00 1.00 0.97 1.00; 1.00 1.00 0.86 0.93 1.00; 0.55 0.38 0.47 0.50 0.23; 0.04 0.19 0.13 0.17 0.09; 0.07 0.00 0.04 0.00 0.10; 0.00 0.04 0.00 0.00 0.00; 0.00 0.00 0.00 0.05 0.00], ...  % Source-guided
 [1.00 1.00 1.00 1.00 0.88; 1.00 1.00 0.95 0.95 0.83; 1.00 1.00 1.00 0.95 1.00; 0.79 0.73 0.86 0.80 0.89; 0.22 0.31 0.13 0.29 0.27; 0.08 0.04 0.00 0.05 0.00; 0.00 0.00 0.00 0.00 0.09; 0.00 0.00 0.00 0.00 0.00] };    % RL-only

groups = {[1 2], [3 4]};
gtitle = {'Model-guided 与 Estimate-guided', 'Source-guided 与 RL-only'};
for g = 1:numel(groups)
    gv = groups{g};
    figure('Color', 'w', 'Position', [60+40*g 70 940 430], 'Name', sprintf('C1 斜坡存活率 (%d/2)', g));
    axh = gobjects(1,numel(gv));
    for kk = 1:numel(gv)
        vi = gv(kk); M = Mall{vi};
        ax = subplot(1, numel(gv), kk); axh(kk) = ax; imagesc(ax, loadCtr, slopes, M, [0 1]); set(ax, 'YDir', 'normal');
        colormap(ax, parula); xlabel(ax, 'Load [kg]'); ylabel(ax, 'Slope [deg]');
        title(ax, variants{vi}, 'FontWeight', 'bold');
        for si = 1:numel(slopes); for bi = 1:numel(loadCtr)
            if ~isnan(M(si, bi)); text(ax, loadCtr(bi), slopes(si), sprintf('%.2f', M(si, bi)), ...
                'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', (M(si,bi)<0.5)*[1 1 1]); end
        end; end
    end
    set(axh(1), 'Position', [0.07 0.14 0.37 0.74]);
    set(axh(2), 'Position', [0.50 0.14 0.37 0.74]);
    colorbar(axh(2), 'Position', [0.905 0.14 0.022 0.74]);
    sgtitle(sprintf('%s 在不同坡度与负载下的抗倾覆成功率', gtitle{g}), 'FontWeight', 'bold');
end
end
