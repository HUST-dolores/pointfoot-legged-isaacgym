function A1_multiseed()
% A1(多seed版)负载质量估计精度:各方案 mass-RMSE 的 3-seed 均值±标准差 + 逐 seed 散点。
% 【独立脚本】数据写死(不读 play 文件),配色/字体自带;不依赖也不修改原 A1_estimation_accuracy.m。
% 每 seed 的 RMSE = 平地行走 vx0.5、load2-30 下逐 env 末段(估计均值−真值)的 RMSE(口径同 notebook §8.6)。
% 生成于 2026-06-14(§8.6 四变体各 3 seed 定稿)。数值与 §8.6 逐 seed 表一致。末尾另存 PDF。

set(groot,'defaultAxesFontName','Noto Sans CJK SC','defaultTextFontName','Noto Sans CJK SC');
variants = {'Model-guided','Estimate-guided','Source-guided','RL-only'};
col = {[0.902 0.624 0.000],[0.000 0.620 0.451],[0.337 0.706 0.914],[0.835 0.369 0.000]};
% 逐 seed mass-RMSE [kg](行=变体,列=seed),来自 §8.6
rmse = [1.12 1.65 1.98;     % Model    (s1 s2 s3)
        1.20 5.49 2.50;     % Estimate (s1 s2 s3)
        3.01 2.64 3.65;     % Source   (s42 s2 s3)
        3.63 7.46 7.59];    % RL-only  (s1 s2 s3)
mu = mean(rmse,2); sd = std(rmse,1,2);   % 总体标准差(/N,与 numpy 默认一致)

figure('Color','w','Position',[80 80 680 480],'Name','A1 多seed 估计精度'); hold on; grid on; box on;
for i = 1:4
    bar(i, mu(i), 0.6, 'FaceColor', col{i}, 'FaceAlpha',0.85, 'EdgeColor','k');
end
errorbar(1:4, mu, sd, 'k', 'LineStyle','none', 'LineWidth',1.4, 'CapSize',12);
for i = 1:4   % 逐 seed 散点(轻微横向抖动)
    jit = (rand(1,3)-0.5)*0.18;
    scatter(i+jit, rmse(i,:), 42, 'MarkerFaceColor','w','MarkerEdgeColor',col{i}*0.6,'LineWidth',1.2);
end
set(gca,'XTick',1:4,'XTickLabel',variants); xlim([0.4 4.6]); ylim([0 max(rmse(:))*1.12]);
ylabel('负载质量估计 RMSE [kg](越小越好)');
title('各方案负载质量估计精度(3 seed 均值 ± 标准差,空心点=各 seed)','FontWeight','bold');
% 标注均值
for i=1:4; text(i, mu(i)+sd(i)+0.25, sprintf('%.2f',mu(i)), 'HorizontalAlignment','center','FontWeight','bold'); end
exportgraphics(gcf, fullfile(fileparts(mfilename('fullpath')),'paper_pdf','A1_多seed_质量估计精度.pdf'), 'ContentType','vector');
end
