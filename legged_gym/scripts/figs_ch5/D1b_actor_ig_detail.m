function D1b_actor_ig_detail()
% D1b actor-IG 细分归因热图:所有输入组(coarse) × 四变体,颜色=归因百分比。附录级细节。
% 数据:各 run 的 actor_ig/actor_ig_summary_*.csv(target=all, coarse)。一张图。不自动保存。

here = fileparts(mfilename('fullpath'));
BASE = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A');
runs = { 'Jun04_23-18-34_wide2-30_model_guided_seed1',    'Model-guided'; ...
         'Jun06_00-18-09_wide2-30_estimate_guided_seed1', 'Estimate-guided'; ...
         'Jun05_17-50-01_wide2-30_source_guided_seed42',  'Source-guided'; ...
         'Jun05_03-01-38_wide2-30_rl_only_seed1',         'RL-only' };

% 收集所有出现过的 group(保持顺序)
allgroups = {}; perrun = cell(size(runs,1),1);
for vi = 1:size(runs,1)
    d = dir(fullfile(BASE, runs{vi,1}, 'actor_ig', 'actor_ig_summary_*.csv'));
    if isempty(d); continue; end
    [~,ix]=max([d.datenum]); T=readtable(fullfile(d(ix).folder,d(ix).name),'TextType','string');
    sel=(T.target=="all")&(T.groupset=="coarse");
    g=T.group(sel); p=T.percent(sel); m=containers.Map();
    for k=1:numel(g); m(char(g(k)))=p(k); if ~any(strcmp(allgroups,char(g(k)))); allgroups{end+1}=char(g(k)); end; end %#ok<AGROW>
    perrun{vi}=m;
end
M = nan(numel(allgroups), size(runs,1));
for vi=1:size(runs,1)
    if isempty(perrun{vi}); continue; end
    for gi=1:numel(allgroups)
        if isKey(perrun{vi},allgroups{gi}); M(gi,vi)=perrun{vi}(allgroups{gi}); end
    end
end

figure('Color','w','Position',[80 60 640 560],'Name','D1b actor-IG 细分');
imagesc(M); colormap(parula); cb=colorbar; cb.Label.String='Attribution [%]';
set(gca,'YTick',1:numel(allgroups),'YTickLabel',strrep(allgroups,'_','\_'));
set(gca,'XTick',1:size(runs,1),'XTickLabel',runs(:,2));
for gi=1:numel(allgroups); for vi=1:size(runs,1)
    if ~isnan(M(gi,vi)); text(vi,gi,sprintf('%.1f',M(gi,vi)),'HorizontalAlignment','center','FontSize',8,'Color',(M(gi,vi)>30)*[1 1 1]); end
end; end
title({'D1b Actor-IG attribution by input group (detail)','load 2-30kg, flat, command 0.5'}, 'FontWeight','bold');
end
