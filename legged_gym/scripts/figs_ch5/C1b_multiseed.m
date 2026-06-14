function C1b_multiseed()
% C1b(多seed版)抗倾覆成功率 vs 坡度,对 2–30kg 负载平均,4 变体 3-seed 均值±标准差。
% 【独立脚本】数据写死(不读 play 文件),配色/字体自带;不依赖也不修改原 C1b_survival_vs_slope.m。
% 数据 = 四变体各 3 seed 的 walk-vx0.5 斜坡 battery(100env,load2-30,flat+倾斜重力,基线 seed 选 100-env 正本),
%   每 seed 先对所有载荷 env 求"末5s 直立(|pitch|<25 且 |roll|<25,占比≥0.8)"的均值,再跨 seed 求 mean±std。
% 生成于 2026-06-14(notebook §8.6 四变体各 3 seed 定稿;24° 均值与 §8.6 表一致:Model .57/Est .45/Src .21/RL .02)。
%
% 逐 seed 原始(survival per slope,行=seed,列=[8 12 16 20 22 24 26 28]°)备查:
%  Model  : [1.0 1.0 .996 .989 .910 .709 .278 .075; .99 .99 .979 .835 .819 .247 .098 .010; 1.0 1.0 1.0 .908 .847 .765 .073 .010]
%  Estimate:[1.0 1.0 1.0 .969 .857 .711 .410 .010; 1.0 .96 .890 .700 .440 .100 .000 .000; 1.0 1.0 1.0 .960 .823 .532 .083 .020]
%  Source : [1.0 .99 .957 .448 .119 .036 .012 .010; 1.0 1.0 1.0 .952 .723 .523 .196 .050; 1.0 .99 .979 .375 .235 .085 .011 .000]
%  RL-only: [.99 .959 .989 .816 .234 .041 .011 .000; .99 .98 .810 .240 .130 .020 .000 .000; .98 .98 .880 .460 .091 .000 .000 .000]

set(groot,'defaultAxesFontName','Noto Sans CJK SC','defaultTextFontName','Noto Sans CJK SC');
slopes = [8 12 16 20 22 24 26 28];
variants = {'Model-guided','Estimate-guided','Source-guided','RL-only'};
col = {[0.902 0.624 0.000],[0.000 0.620 0.451],[0.337 0.706 0.914],[0.835 0.369 0.000]};
% 3-seed 均值 / 标准差(对负载平均的抗倾覆成功率)
mu = [0.997 0.997 0.992 0.911 0.859 0.574 0.150 0.032;
      1.000 0.987 0.963 0.876 0.707 0.448 0.164 0.010;
      1.000 0.993 0.979 0.592 0.359 0.215 0.073 0.020;
      0.987 0.973 0.893 0.505 0.152 0.020 0.004 0.000];
sd = [0.005 0.005 0.009 0.063 0.038 0.232 0.091 0.031;
      0.000 0.019 0.052 0.125 0.189 0.256 0.177 0.008;
      0.000 0.005 0.018 0.257 0.262 0.219 0.087 0.022;
      0.005 0.010 0.074 0.237 0.060 0.017 0.005 0.000];

figure('Color','w','Position',[80 80 740 500],'Name','C1b 多seed 抗倾覆vs坡度'); hold on; grid on; box on;
yline(0.5,'k--','HandleVisibility','off');
h = gobjects(1,4);
for i = 1:4
    lo = max(mu(i,:)-sd(i,:),0); hi = min(mu(i,:)+sd(i,:),1);
    fill([slopes fliplr(slopes)],[hi fliplr(lo)],col{i},'FaceAlpha',0.13,'EdgeColor','none','HandleVisibility','off');
    h(i) = plot(slopes,mu(i,:),'-o','Color',col{i},'LineWidth',2,'MarkerFaceColor',col{i},'MarkerSize',5);
end
xlabel('坡度 [\circ]'); ylabel('抗倾覆成功率(对 2–30 kg 负载平均)'); ylim([0 1.04]); xlim([7 29]);
legend(h,variants,'Location','southwest');
title('对各负载平均的抗倾覆成功率随坡度的变化(3 seed 均值 ± 标准差)','FontWeight','bold');
exportgraphics(gcf, fullfile(fileparts(mfilename('fullpath')),'paper_pdf','C1b_多seed_抗倾覆vs坡度.pdf'), 'ContentType','vector');
end
