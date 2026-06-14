function A2_estimate_timeseries(target_load, tlim)
% A2 估计时域对比:每策略一张图(3 子图 mass/comx/comy),真值(黑)+ model-based 解析(灰虚)+ encoder(变体色)。
% 各策略文件每-env 真值不同 → 统一挑"真实负载最接近 target_load 的 env",使四张图同负载、可比。
% 用法:A2_estimate_timeseries(15, [1 20])  默认 target=15kg、时间窗 1–20s。不自动保存。
if nargin<1 || isempty(target_load); target_load = 26; end
if nargin<2 || isempty(tlim); tlim = [1 20]; end

here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
variants = {'Model-guided', 'Estimate-guided', 'Source-guided', 'RL-only'};
col = ch5_colors();
qsCol = [0.45 0.45 0.45];                 % model-based 解析:灰
rows = { {'payload_mass_ref_all','payload_mass_all','encoder_mass_all','Load mass [kg]'}; ...
         {'true_com_delta_x_all','modelC_com_delta_x_all','encoder_com_delta_x_all','CoM x [m]'}; ...
         {'true_com_delta_y_all','modelC_com_delta_y_all','encoder_com_delta_y_all','CoM y [m]'} };

% 每个策略单独一张图(3 子图:mass/comx/comy),每张都带图例
for ci = 1:numel(variants)
    [S, ok] = pick_file(ED, variants{ci});
    figure('Color','w','Position',[60+30*ci 60 560 720],'Name',['A2 ' variants{ci}]);
    for ri = 1:3
        ax = subplot(3, 1, ri); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
        if ~ok; title(ax,'(无数据)'); continue; end
        spec = rows{ri};
        [t, j, L] = deal_env(S, target_load);
        h2 = plot(ax, t, getcol(S,spec{2},j), '--', 'Color', qsCol, 'LineWidth', 1.3);     % model-based 解析
        h3 = plot(ax, t, getcol(S,spec{3},j), '-',  'Color', col{ci}, 'LineWidth', 1.8);   % encoder
        h1 = plot(ax, t, getcol(S,spec{1},j), 'k',  'LineWidth', 2.4);                     % 真值(画最上层,始终可见)
        ylabel(ax, spec{4}); xlim(ax, tlim);
        if ri==1; ylim(ax, [-10 60]); end                 % mass 子图四图统一纵轴
        if ri==3; xlabel(ax, 'Time [s]'); end
        legend(ax, [h1 h2 h3], {'real (ground truth)', 'model-based (analytic)', [variants{ci} ' encoder']}, 'Location','best');
    end
    sgtitle(sprintf('固定 %.0f kg 负载下 %s 的负载质量与质心估计时间序列', L, variants{ci}), 'FontWeight','bold');
end
end

function [S, ok] = pick_file(ED, vname)
% 固定 26kg 同步循环负载数据(_load26-26_cycle*),四策略严格同负载;无则回退恒载范围数据。
S=[]; ok=false; best=''; bt=-1;
pat = 'play_data_*walk_vx0.5*_load26-26_flat_cycle*.mat';
if isempty(dir(fullfile(ED, pat))); pat = 'play_data_*walk_vx0.5*_load2-30_flat_cycle*.mat'; end
if isempty(dir(fullfile(ED, pat))); pat = 'play_data_*walk_vx0.5*_load2-30_flat.mat'; end
for f = dir(fullfile(ED, pat))'
    b=f.name; if contains(b,'slope')||contains(b,'estop')||contains(b,'_corr'); continue; end
    Q = load(fullfile(f.folder,b));
    if ~isfield(Q,'meta')||~contains(char(getf(Q.meta,'load_run','')),'wide2-30'); continue; end
    if ~strcmp(variant_name(Q.meta), vname); continue; end
    if f.datenum>bt; bt=f.datenum; best=fullfile(f.folder,b); end
end
if isempty(best); return; end
S=load(best); ok=true;
end

function [t, j, L] = deal_env(S, target)
% 统一挑"真实负载最接近 target、且负载在体上够久"的 env,使各策略图同负载、可比。
dt=0.02; if isfield(S,'dt')&&~isempty(S.dt); dt=double(S.dt(1)); end
ref = orient(S.payload_mass_ref_all); [T,N]=size(ref);
if isfield(S,'load_on_body_all'); dur = sum(orient(S.load_on_body_all)>0.5, 1);
else; dur = sum(ref>0.3, 1); end
Lj = nan(1,N);
for k=1:N; on=ref(:,k)>0.3; if any(on); Lj(k)=median(ref(on,k),'omitnan'); end; end
valid = (dur > 0.5*T) & ~isnan(Lj);          % 负载在体>一半时间
if ~any(valid); valid = ~isnan(Lj); end       % 退化:只要有负载
cand = find(valid); [~,ix] = min(abs(Lj(cand)-target)); j = cand(ix);
L = Lj(j); t=(0:T-1)*dt;
end

function y = getcol(S, field, j)
A = orient(S.(field)); j=min(j,size(A,2)); y=A(:,j);
end
function A = orient(A); A = double(A); if size(A,1)<size(A,2); A=A.'; end; end
function vn = variant_name(meta)
qs=logical(getf(meta,'use_qs_in_obs',true)); re=logical(getf(meta,'use_load_residual_estimation',false)); tq=logical(getf(meta,'use_torques_in_obs',true));
if qs&&re; vn='Model-guided'; elseif ~qs&&~re&&~tq; vn='RL-only'; elseif qs&&~re; vn='Estimate-guided'; else; vn='Source-guided'; end
end
function v = getf(s,f,d); if isfield(s,f)&&~isempty(s.(f)); v=s.(f); else; v=d; end; end
