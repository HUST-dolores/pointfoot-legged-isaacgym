% make_all.m — 一键出第X章 co-design 全部图 (输出到 figs_chx/out/)
% 前置: 已有 data/codesign_figs_data.mat (由 prep_data.py 生成)。
% 需要 MATLAB R2020a+ (exportgraphics); 更早版本自动回退 print -dpdf。
% 若中文显示为方块: 在 MATLAB 里 set(groot,'defaultAxesFontName','<系统CJK字体>')。
here = fileparts(mfilename('fullpath')); addpath(here);
fprintf('出图中...\n');
fig1_bo_convergence();
fig2_table2_bars();
fig3_finetune_curve();
fig4_scenario_vs_motor();
fig5_scenario_landscape();
fig6_foot_height();
fig7_robustness();
fprintf('完成 -> %s\n', fullfile(here, 'out'));
