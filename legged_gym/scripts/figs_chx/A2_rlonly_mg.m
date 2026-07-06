function A2_rlonly_mg(target_load, tlim)
% 时域负载估计对比 (新策略): rlonly 与 mg_e1(model-guided) 各一张图。
% 每图 3 子图(负载质量 / CoM x / CoM y),三条线:
%   真值(黑) + model-based 解析(灰虚) + encoder 最终估计(策略色)。
% 数据: exported/play_data_*A2_rlonly.mat 与 *A2_mg.mat(滚动 vx0.5 + 15kg 循环负载, 平坦)。
% 用法: A2_rlonly_mg           % 默认 target=15kg, 全 40s
%       A2_rlonly_mg(15,[1 40])
% (基于 figs_ch5/A2_estimate_timeseries.m 适配到新的 rlonly / mg_e1 数据)
if nargin < 1 || isempty(target_load); target_load = 15; end
if nargin < 2 || isempty(tlim); tlim = [1 40]; end
here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
qsCol = [0.45 0.45 0.45];
% {文件tag, 图例名, encoder线色}
specs = { 'rlonly', 'RL-only',      [0.85 0.33 0.10]; ...   % 橙
          'mg',     'Model-guided', [0.00 0.45 0.74] };      % 蓝
rows = { {'payload_mass_ref_all','payload_mass_all','encoder_mass_all','Load mass [kg]'}; ...
         {'true_com_delta_x_all','modelC_com_delta_x_all','encoder_com_delta_x_all','CoM x [m]'}; ...
         {'true_com_delta_y_all','modelC_com_delta_y_all','encoder_com_delta_y_all','CoM y [m]'} };
for ci = 1:size(specs,1)
    fn = dir(fullfile(ED, ['play_data_*A2_' specs{ci,1} '.mat']));
    if isempty(fn); warning('无 %s 数据(exported里没有 A2_%s)', specs{ci,1}, specs{ci,1}); continue; end
    [~, ix] = max([fn.datenum]);                 % 用最新一份
    S = load(fullfile(fn(ix).folder, fn(ix).name));
    [t, j, L] = deal_env(S, target_load);
    figure('Color','w','Position',[60+40*ci 60 720 740],'Name',['A2 ' specs{ci,2}]);
    for ri = 1:3
        ax = subplot(3,1,ri); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
        spec = rows{ri};
        h2 = plot(ax, t, getcol(S,spec{2},j), '--', 'Color', qsCol,      'LineWidth', 1.3);  % model-based 解析
        h3 = plot(ax, t, getcol(S,spec{3},j), '-',  'Color', specs{ci,3}, 'LineWidth', 1.8);  % encoder
        h1 = plot(ax, t, getcol(S,spec{1},j), 'k',                        'LineWidth', 2.4);   % 真值(最上层)
        ylabel(ax, spec{4}); xlim(ax, tlim);
        if ri==1; ylim(ax, [-5 25]); end
        if ri==3; xlabel(ax, 'Time [s]'); end
        legend(ax, [h1 h2 h3], {'real (ground truth)', 'model-based (analytic)', [specs{ci,2} ' encoder']}, 'Location','best', 'FontSize',8);
    end
    sgtitle(sprintf('%s  —  负载质量/质心 估计时域（%.0fkg 循环负载, 滚动 vx0.5）', specs{ci,2}, L), 'FontWeight','bold', 'FontSize', 11);
    try, cd_export(gcf, ['A2_timeseries_' specs{ci,1}]); catch, end
end
end

% ---- 以下辅助函数复制自 figs_ch5/A2_estimate_timeseries.m ----
function [t, j, L] = deal_env(S, target)
% 挑"真实负载最接近 target、且负载在体上够久"的 env。
dt = 0.02; if isfield(S,'dt')&&~isempty(S.dt); dt=double(S.dt(1)); end
ref = orient(S.payload_mass_ref_all); [T,N] = size(ref);
if isfield(S,'load_on_body_all'); dur = sum(orient(S.load_on_body_all)>0.5, 1);
else; dur = sum(ref>0.3, 1); end
Lj = nan(1,N);
for k=1:N; on=ref(:,k)>0.3; if any(on); Lj(k)=median(ref(on,k),'omitnan'); end; end
valid = (dur > 0.3*T) & ~isnan(Lj);
if ~any(valid); valid = ~isnan(Lj); end
cand = find(valid); [~,ix] = min(abs(Lj(cand)-target)); j = cand(ix);
L = Lj(j); t = (0:T-1)*dt;
end
function y = getcol(S, field, j); A = orient(S.(field)); j=min(j,size(A,2)); y=A(:,j); end
function A = orient(A); A = double(A); if size(A,1)<size(A,2); A=A.'; end; end
