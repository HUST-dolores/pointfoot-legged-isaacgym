function A2b_estimation_rmse_timeseries(tlim, show_band)
% A2b 估计 RMSE vs 时间(正文版)。3 子图:mass / CoM x / CoM y。
% 每方法一条 RMSE(t)=sqrt(mean_i (est_i(t)-true_i(t))^2)(对所有带载 env);阴影=env 间 25–75% |误差|。
% 方法:4 策略 encoder(变体色)。直接回答"是否收敛、稳态误差多大"。
% 用法:A2b_estimation_rmse_timeseries([1 20], true)。不自动保存。
if nargin<1||isempty(tlim); tlim=[1 20]; end
if nargin<2||isempty(show_band); show_band=true; end

here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
variants = {'Model-guided', 'Estimate-guided', 'Source-guided', 'RL-only'};
col = ch5_colors();
rows = { {'payload_mass_ref_all','encoder_mass_all','payload_mass_all','RMSE load mass [kg]'}; ...
         {'true_com_delta_x_all','encoder_com_delta_x_all','modelC_com_delta_x_all','RMSE CoM x [cm]'}; ...
         {'true_com_delta_y_all','encoder_com_delta_y_all','modelC_com_delta_y_all','RMSE CoM y [cm]'} };

files = cell(1,numel(variants));
for ci=1:numel(variants); files{ci} = pick_file(ED, variants{ci}); end
% 单图、四策略 encoder RMSE(t) 叠加 → 横向对比"谁估得最准/收敛最快"(结论导向)
figure('Color','w','Position',[60 50 760 760],'Name','A2b 四策略估计RMSE对比');
for ri = 1:3
    ax = subplot(3,1,ri); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    spec = rows{ri}; h=gobjects(0); lab={};
    if ri==1; sc=1; unit='kg'; else; sc=100; unit='cm'; end   % CoM 子图 m→cm
    for ci = 1:numel(variants)
        if isempty(files{ci}); continue; end
        [rt, rmse, q1, q3] = rmse_curve(files{ci}, spec{1}, spec{2});
        rmse=rmse*sc; q1=q1*sc; q3=q3*sc;                      % 缩放(mass×1, CoM×100)
        if show_band; band_fill(ax, rt, q1, q3, col{ci}); end
        inwin = isfinite(rmse) & rt>=tlim(1) & rt<=tlim(2);
        mu = mean(rmse(inwin), 'omitnan');                 % 时窗内 RMSE(t) 均值(越小越好)
        h(end+1)=plot(ax, rt, rmse, '-', 'Color', col{ci}, 'LineWidth', 2.2); %#ok<AGROW>
        lab{end+1}=sprintf('%s  (mean %.2f %s)', variants{ci}, mu, unit); %#ok<AGROW>
    end
    ylabel(ax, spec{4}); xlim(ax, tlim);
    if ri==1; ylim(ax,[0 20]); else; ylim(ax,[0 6]); end   % mass 0–20kg, CoM 0–6cm
    if ri==3; xlabel(ax, 'Time [s]'); end
    legend(ax, h, lab, 'Location','northeast');
end
sgtitle('各方案负载估计的 RMSE 随时间的变化', 'FontWeight','bold');
end

% ---- 计算 RMSE(t) 与 25/75 分位带 ----
function [t, rmse, q1, q3] = rmse_curve(S, truefield, estfield)
ref = orient(S.payload_mass_ref_all);
tru = orient(S.(truefield)); est = orient(S.(estfield));
dt=0.02; if isfield(S,'dt')&&~isempty(S.dt); dt=double(S.dt(1)); end
[T,N]=size(ref); t=(0:T-1)*dt;
err = est - tru;   % 全程所有 env(含 off 段,此时真值=0,看卸载是否归零)
rmse=nan(1,T); q1=nan(1,T); q3=nan(1,T);
for k=1:T
    e = err(k,:); e = e(isfinite(e));
    if numel(e) < 2; continue; end
    rmse(k) = sqrt(mean(e.^2));
    ae = abs(e); q1(k)=pctl(ae,25); q3(k)=pctl(ae,75);
end
end

function band_fill(ax, t, q1, q3, c)
m = isfinite(q1) & isfinite(q3);
if ~any(m); return; end
tt=t(m); a=q1(m); b=q3(m);
fill(ax, [tt fliplr(tt)], [a fliplr(b)], c, 'FaceAlpha', 0.12, 'EdgeColor','none', 'HandleVisibility','off');
end

function q = pctl(v, p)   % 手写分位数(不依赖 Stats Toolbox)
v = sort(v(~isnan(v))); n=numel(v);
if n==0; q=nan; return; end
if n==1; q=v(1); return; end
idx=(p/100)*(n-1)+1; lo=floor(idx); hi=ceil(idx); f=idx-lo;
q = v(lo)*(1-f) + v(hi)*f;
end

function S = pick_file(ED, vname)
% 优先用同步循环负载数据(_cycle*),体现加/卸载瞬态;无则回退恒载 walk-flat。
S=[]; best=''; bt=-1;
pat = 'play_data_*walk_vx0.5*_load2-30_flat_cycle*.mat';
if isempty(dir(fullfile(ED, pat))); pat = 'play_data_*walk_vx0.5*_load2-30_flat.mat'; end
for f = dir(fullfile(ED, pat))'
    b=f.name; if contains(b,'slope')||contains(b,'estop')||contains(b,'_corr'); continue; end
    Q = load(fullfile(f.folder,b));
    if ~isfield(Q,'meta')||~contains(char(getf(Q.meta,'load_run','')),'wide2-30'); continue; end
    if ~strcmp(variant_name(Q.meta), vname); continue; end
    if f.datenum>bt; bt=f.datenum; best=fullfile(f.folder,b); end
end
if ~isempty(best); S=load(best); end
end
function A = orient(A); A = double(A); if size(A,1)<size(A,2); A=A.'; end; end
function vn = variant_name(meta)
qs=logical(getf(meta,'use_qs_in_obs',true)); re=logical(getf(meta,'use_load_residual_estimation',false)); tq=logical(getf(meta,'use_torques_in_obs',true));
if qs&&re; vn='Model-guided'; elseif ~qs&&~re&&~tq; vn='RL-only'; elseif qs&&~re; vn='Estimate-guided'; else; vn='Source-guided'; end
end
function v = getf(s,f,d); if isfield(s,f)&&~isempty(s.(f)); v=s.(f); else; v=d; end; end
