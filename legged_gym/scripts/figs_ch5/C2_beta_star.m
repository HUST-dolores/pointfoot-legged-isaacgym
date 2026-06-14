function C2_beta_star()
% C2(C1 的补充)抗倾覆边界 β*(load):末态直立率随坡度跌破 0.5 的临界坡度,vs 负载,四变体。
% 【数据已写死(2026-06-14)】baseline seed(Model/Estimate/RL=seed1,Source=seed42)的逐负载箱 β*;
%   口径同原版(walk_vx0.5, load2-30, 100-env 斜坡 battery,坡度 [8 12 16 20 22 24 26 28]°);
%   不再读 play 文件,防数据移走/丢失后失效。配色/字体内联(原 ch5_colors 值),零外部依赖。不自动保存。
% 多 seed 版见 C2_multiseed.m。
variants = {'Model-guided', 'Estimate-guided', 'Source-guided', 'RL-only'};
col = {[0.902 0.624 0.000],[0.000 0.620 0.451],[0.337 0.706 0.914],[0.835 0.369 0.000]};
set(groot,'defaultAxesFontName','Noto Sans CJK SC','defaultTextFontName','Noto Sans CJK SC');
loadEdges = [2 8 14 20 26 30.1]; loadCtr = (loadEdges(1:end-1)+loadEdges(2:end))/2;  % [5 11 17 23 28] kg
% β* per load-bin(行=变体,列中心=loadCtr)
B = [25.33 25.57 24.39 24.61 25.02;
     25.86 25.34 25.14 25.00 23.78;
     20.18 19.25 19.70 20.00 18.60;
     21.01 21.11 20.98 21.17 21.27];
nv = numel(variants);
figure('Color','w','Position',[80 80 720 480],'Name','C2 抗倾覆边界 β*'); hold on; grid on; box on;
h = gobjects(1,nv);
for vi=1:nv
    h(vi) = plot(loadCtr, B(vi,:), '-o', 'Color', col{vi}, 'LineWidth', 2, 'MarkerFaceColor', col{vi});
end
xlabel('Load [kg]'); ylabel('Tipping boundary \beta^* [deg]');
legend(h, variants, 'Location', 'southwest');
title({'C2 Anti-tipping boundary \beta^*(load)', 'critical slope where terminal upright-rate < 0.5'}, 'FontWeight','bold');
ylim([16 28]);
for vi=1:nv; fprintf('  %-16s beta* mean = %.1f deg\n', variants{vi}, mean(B(vi,:),'omitnan')); end
end
