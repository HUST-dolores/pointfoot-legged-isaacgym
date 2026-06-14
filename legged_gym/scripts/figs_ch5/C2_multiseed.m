function C2_multiseed()
% C2(多seed版)抗倾覆边界 β* vs 负载,4 变体 3-seed 均值±标准差。
% 【独立脚本】数据写死(不读 play 文件),配色/字体自带;不依赖也不修改原 C2_beta_star.m。
% β* = 该负载箱"末态直立率跌破 0.5"的临界坡度(在 [8,12,16,20,22,24,26,28]° 上线性插值);
%   逐 seed 逐箱算 β*,再跨 seed 求 mean±std(基线 seed 选 100-env 正本,无 NaN)。
% 生成于 2026-06-14(notebook §8.6 四变体各 3 seed 定稿)。
%
% 逐 seed 原始(β* per load-bin,行=seed,列中心=[5 11 17 23 28]kg)备查:
%  Model  : [25.33 25.57 24.39 24.61 25.02; 23.04 23.07 23.02 23.35 22.93; 25.35 24.67 24.72 24.46 24.45]
%  Estimate:[25.86 25.34 25.14 25.00 23.78; 23.03 22.18 20.49 21.29 20.76; 23.59 24.39 24.58 24.10 23.62]
%  Source : [20.18 19.25 19.70 20.00 18.60; 24.53 24.73 24.57 23.39 22.16; 19.64 18.91 20.13 18.57 18.25]
%  RL-only: [21.01 21.11 20.98 21.17 21.27; 20.00 18.25 18.01 17.74 16.99; 20.75 20.62 19.15 18.14 17.45]

set(groot,'defaultAxesFontName','Noto Sans CJK SC','defaultTextFontName','Noto Sans CJK SC');
ctr = [5 11 17 23 28];   % 负载箱中心 [kg]
variants = {'Model-guided','Estimate-guided','Source-guided','RL-only'};
col = {[0.902 0.624 0.000],[0.000 0.620 0.451],[0.337 0.706 0.914],[0.835 0.369 0.000]};
% 3-seed 均值 / 标准差
mu = [24.57 24.44 24.04 24.14 24.13;
      24.16 23.97 23.40 23.46 22.72;
      21.45 20.96 21.47 20.65 19.67;
      20.59 19.99 19.38 19.02 18.57];
sd = [1.08 1.03 0.74 0.56 0.88;
      1.22 1.32 2.07 1.58 1.39;
      2.19 2.67 2.20 2.02 1.77;
      0.43 1.25 1.22 1.53 1.92];

figure('Color','w','Position',[80 80 740 500],'Name','C2 多seed beta* vs 负载'); hold on; grid on; box on;
h = gobjects(1,4);
for i = 1:4
    lo = mu(i,:)-sd(i,:); hi = mu(i,:)+sd(i,:);
    fill([ctr fliplr(ctr)],[hi fliplr(lo)],col{i},'FaceAlpha',0.13,'EdgeColor','none','HandleVisibility','off');
    h(i) = plot(ctr,mu(i,:),'-o','Color',col{i},'LineWidth',2,'MarkerFaceColor',col{i},'MarkerSize',5);
end
xlabel('负载质量 [kg]'); ylabel('抗倾覆边界 \beta^* [\circ]'); ylim([16 28]); xlim([3 30]);
legend(h,variants,'Location','southwest');
title('抗倾覆边界 \beta^* 随负载的变化(3 seed 均值 ± 标准差)','FontWeight','bold');
exportgraphics(gcf, fullfile(fileparts(mfilename('fullpath')),'paper_pdf','C2_多seed_beta_star.pdf'), 'ContentType','vector');
end
