function A1c_com_accuracy()
% A1c 负载质心偏置(CoM)估计精度:encoder_com_delta vs true_com_delta(末段per-env),x/y 双子图,四变体。
% 观点:不止质量,质心偏置也估得准(呼应 C4 偏心实验)。一张图。不自动保存。

here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
variants = {'Model-guided', 'Estimate-guided', 'Source-guided', 'RL-only'};
col = ch5_colors();
axes_ = {'x','y'};

figure('Color','w','Position',[60 80 1000 470],'Name','A1c CoM 估计精度');
for ai = 1:2
    ax = subplot(1,2,ai); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    plot(ax,[-0.2 0.2],[-0.2 0.2],'k--');
    h = gobjects(1,numel(variants)); lab = cell(1,numel(variants));
    for vi = 1:numel(variants)
        [tru, est] = collect(ED, variants{vi}, axes_{ai});
        if isempty(tru); h(vi)=plot(ax,nan,nan,'o','Color',col{vi}); lab{vi}=variants{vi}; continue; end
        scatter(ax, tru, est, 22, col{vi}, 'filled', 'MarkerFaceAlpha', 0.55, 'MarkerEdgeColor', col{vi}, 'MarkerEdgeAlpha', 0.8, 'HandleVisibility','off');
        rmse = sqrt(mean((est-tru).^2,'omitnan'));
        h(vi) = plot(ax,nan,nan,'o','Color',col{vi},'MarkerFaceColor',col{vi});
        lab{vi} = sprintf('%s (RMSE %.3f m)', variants{vi}, rmse);
    end
    xlabel(ax,sprintf('True CoM offset %s [m]',axes_{ai})); ylabel(ax,sprintf('Estimated CoM %s [m]',axes_{ai}));
    title(ax, sprintf('CoM-%s', axes_{ai}), 'FontWeight','bold');
    axis(ax,[-0.2 0.2 -0.2 0.2]); axis(ax,'square');   % x/y 轴同范围,单位线为真45°、两子图可比
    legend(ax,h,lab,'Location','northwest');   % 两个子图各自显示图注(RMSE 按各自轴)
end
sgtitle('2–30 kg 负载范围内各方案对负载质心 x、y 向偏移的估计精度', 'FontWeight','bold');
end

function [tru, est] = collect(ED, vname, ax)
tru=[]; est=[]; best=''; bt=-1;
for f = dir(fullfile(ED, 'play_data_*walk_vx0.5*_load2-30_flat.mat'))'
    if contains(f.name,'slope')||contains(f.name,'estop'); continue; end
    S = load(fullfile(f.folder,f.name));
    if ~isfield(S,'meta')||~contains(char(getf(S.meta,'load_run','')),'wide2-30'); continue; end
    if ~strcmp(variant_name(S.meta), vname); continue; end
    if f.datenum>bt; bt=f.datenum; best=fullfile(f.folder,f.name); end
end
if isempty(best); return; end
S=load(best);
E = orient(S.(['encoder_com_delta_' ax '_all'])); R = orient(S.(['true_com_delta_' ax '_all']));
ref = orient(S.payload_mass_ref_all);
dt=0.02; if isfield(S,'dt')&&~isempty(S.dt); dt=double(S.dt(1)); end
[T,N]=size(E); t=(0:T-1)'*dt; last=t>5.0;
for j=1:N
    on=ref(:,j)>0.3; if ~any(on); continue; end
    tru(end+1)=mean(R(last,j),'omitnan'); est(end+1)=mean(E(last,j),'omitnan'); %#ok<AGROW>
end
end
function A = orient(A); A = double(A); if size(A,1)<size(A,2); A=A.'; end; end
function vn = variant_name(meta)
qs=logical(getf(meta,'use_qs_in_obs',true)); re=logical(getf(meta,'use_load_residual_estimation',false)); tq=logical(getf(meta,'use_torques_in_obs',true));
if qs&&re; vn='Model-guided'; elseif ~qs&&~re&&~tq; vn='RL-only'; elseif qs&&~re; vn='Estimate-guided'; else; vn='Source-guided'; end
end
function v = getf(s,f,d); if isfield(s,f)&&~isempty(s.(f)); v=s.(f); else; v=d; end; end
