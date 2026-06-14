function D1_actor_ig()
% D1 actor-IG 输入归因(机理):四变体 actor 对输入分组的积分梯度归因(target=all, coarse)。
% 聚合成 5 类:Load latent(encoder mass/CoM)/ Explicit model-based obs / Raw torques / Prev actions / Other。
% 注:IG 是局部梯度归因,≠因果必要性;因果载体见 C6(消融)。一张分组柱状图。不自动保存。
% 数据:各 run 的 actor_ig/actor_ig_summary_*.csv(取最新)。

here = fileparts(mfilename('fullpath'));
BASE = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A');
runs = { 'Jun04_23-18-34_wide2-30_model_guided_seed1',    'Model-guided'; ...
         'Jun06_00-18-09_wide2-30_estimate_guided_seed1', 'Estimate-guided'; ...
         'Jun05_17-50-01_wide2-30_source_guided_seed42',  'Source-guided'; ...
         'Jun05_03-01-38_wide2-30_rl_only_seed1',         'RL-only' };
% 拆开 encoder latent(运行时被读)与 explicit model-based obs(冗余),与消融结论一致
cats = {'Load latent','Explicit model-based obs','Raw torques','Prev actions','Other'};
catkeys = { {'est_mass','est_com'}, ...           % policy-facing encoder latent(运行时主用)
            {'qs_load','qs_residual'}, ...        % obs 里的显式 model-based 特征(消融证明冗余)
            {'torque'}, ...
            {'previous_action','prev_action'}, ...
            {'base_ang_vel','projected_gravity','dof_pos','dof_vel','est_lin_vel'} };

M = nan(size(runs,1), numel(cats));   % variant × category percent
for vi = 1:size(runs,1)
    d = dir(fullfile(BASE, runs{vi,1}, 'actor_ig', 'actor_ig_summary_*.csv'));
    if isempty(d); continue; end
    [~,ix] = max([d.datenum]); T = readtable(fullfile(d(ix).folder, d(ix).name), 'TextType','string');
    sel = (T.target=="all") & (T.groupset=="coarse");
    g = T.group(sel); p = T.percent(sel);
    for ci = 1:numel(cats)
        s = false(size(g));
        for kk = catkeys{ci}; s = s | contains(g, kk{1}); end
        M(vi,ci) = sum(p(s));
    end
    fprintf('  %-16s latent=%.1f MBobs=%.1f torq=%.1f prevA=%.1f other=%.1f\n', runs{vi,2}, M(vi,:));
end

cc = ch5_colors(); col = cat(1, cc{:});
figure('Color','w','Position',[80 80 760 470],'Name','D1 actor-IG');
b = bar(M', 'grouped'); for vi=1:numel(b); b(vi).FaceColor = col(vi,:); end
set(gca,'XTickLabel',cats); ylabel('Actor IG attribution [%]'); grid on; box on;
legend(runs(:,2), 'Location','northeast');
title({'D1 Actor-IG attribution by input group (load 2-30kg, flat)', ...
       'Model/Estimate read explicit model-based obs (high IG); Source barely uses raw torques (2.3%). Note: IG\neqcausal — ablation (C6) shows the encoder latent is the load-bearing channel.'}, 'FontWeight','bold');
end
