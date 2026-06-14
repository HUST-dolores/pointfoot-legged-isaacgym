function C1b_survival_vs_slope()
% C1b 末态直立率(抗倾覆成功率)vs 坡度,对 2–30kg 负载平均,四变体。
% 【数据已写死(2026-06-14)】baseline seed(Model/Estimate/RL=seed1,Source=seed42)逐坡度对负载平均的存活率;
%   口径同原版(walk_vx0.5, load2-30, 100-env battery);不再读 play 文件。配色/字体内联,零外部依赖。不自动保存。
% 多 seed 版见 C1b_multiseed.m。
variants = {'Model-guided', 'Estimate-guided', 'Source-guided', 'RL-only'};
col = {[0.902 0.624 0.000],[0.000 0.620 0.451],[0.337 0.706 0.914],[0.835 0.369 0.000]};
set(groot,'defaultAxesFontName','Noto Sans CJK SC','defaultTextFontName','Noto Sans CJK SC');
slopes = [8 12 16 20 22 24 26 28];
% 对负载平均的抗倾覆成功率(行=变体,列=slopes)
S = [1.000 1.000 0.996 0.989 0.910 0.709 0.278 0.075;
     1.000 1.000 1.000 0.969 0.857 0.711 0.410 0.010;
     1.000 0.990 0.957 0.448 0.119 0.036 0.012 0.010;
     0.990 0.959 0.989 0.816 0.234 0.041 0.011 0.000];
figure('Color','w','Position',[80 80 720 480],'Name','C1b 存活率vs坡度'); hold on; grid on; box on;
yline(0.5,'k--','HandleVisibility','off');
h = gobjects(1,numel(variants));
for vi=1:numel(variants)
    h(vi)=plot(slopes, S(vi,:), '-o', 'Color', col{vi}, 'LineWidth', 2, 'MarkerFaceColor', col{vi});
end
xlabel('Slope [deg]'); ylabel('抗倾覆成功率(对 2–30 kg 负载平均)');
legend(h, variants, 'Location','southwest'); ylim([0 1.02]);
title('对各负载平均的抗倾覆成功率随坡度的变化', 'FontWeight','bold');
end
