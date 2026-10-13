function A2_rlonly_mg(target_load, tlim, export_pdf)
% Paper-style payload-estimation time series for RL-only and Model-guided RL.
% Each method produces one mass figure and one planar-CoM position figure.
% Lines follow paper10: true (black), Model-based (light method color),
% and the learned result (method color); all three are solid.
% 数据: exported/play_data_*A2_rlonly.mat 与 *A2_mg.mat(滚动 vx0.5 + 15kg 循环负载, 平坦)。
% 用法: A2_rlonly_mg           % 默认 target=15kg, 0–40s, 仅导出 PNG
%       A2_rlonly_mg(15,[0 40])
%       A2_rlonly_mg(15,[1 40],true)  % 明确要求时才额外导出 PDF
if nargin < 1 || isempty(target_load); target_load = 15; end
if nargin < 2 || isempty(tlim); tlim = [0 40]; end
if nargin < 3 || isempty(export_pdf); export_pdf = false; end
here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
% Canonical colors from experiment_notebook2.md and figs_ch5/ch5_colors.m.
% {file tag, paper method name, learned-estimate color}
specs = { 'rlonly', 'RL-only',         [213  94   0] / 255; ...
          'mg',     'Model-guided RL', [230 159   0] / 255 };
rows = { {'payload_mass_ref_all','payload_mass_all','encoder_mass_all','mass [kg]'}; ...
         {'true_com_delta_x_all','modelC_com_delta_x_all','encoder_com_delta_x_all','CoM x-offset [m]'}; ...
         {'true_com_delta_y_all','modelC_com_delta_y_all','encoder_com_delta_y_all','CoM y-offset [m]'} };
positionYLim = {[-0.3 0.2], [-0.4 0.2]};
for ci = 1:size(specs,1)
    fn = dir(fullfile(ED, ['play_data_*A2_' specs{ci,1} '.mat']));
    if isempty(fn); warning('无 %s 数据(exported里没有 A2_%s)', specs{ci,1}, specs{ci,1}); continue; end
    [~, ix] = max([fn.datenum]);
    S = load(fullfile(fn(ix).folder, fn(ix).name));
    [t, j] = deal_env(S, target_load);
    policyCol = specs{ci,3};
    modelCol = 0.55 * policyCol + 0.45 * [1 1 1];
    legendText = {'true', 'Model-based', specs{ci,2}};

    fMass = figure('Color','w', 'Position',[60+40*ci 80 720 360], ...
        'Name',['Payload mass: ' specs{ci,2}]);
    ax = axes(fMass); hold(ax,'on'); box(ax,'on');
    handles = plot_panel(ax, t, S, rows{1}, j, modelCol, policyCol);
    ylabel(ax, rows{1}{4}); xlabel(ax, 'time [s]');
    xlim(ax, tlim); ylim(ax, [-2 25]); style_axis(ax);
    legend(ax, handles, legendText, 'Location','best', ...
        'FontName','Times New Roman', 'FontSize',8, 'Box','on');
    title(ax, specs{ci,2}, 'FontName','Times New Roman', ...
        'FontWeight','bold', 'FontSize',11);
    try, cd_export(fMass, ['A2_mass_' specs{ci,1}], export_pdf); catch, end

    fPosition = figure('Color','w', 'Position',[100+40*ci 60 560 600], ...
        'Name',['Payload position: ' specs{ci,2}]);
    for pi = 1:2
        ax = subplot(2, 1, pi, 'Parent', fPosition); hold(ax,'on'); box(ax,'on');
        spec = rows{pi+1};
        handles = plot_panel(ax, t, S, spec, j, modelCol, policyCol);
        ylabel(ax, spec{4}); xlim(ax, tlim); ylim(ax, positionYLim{pi});
        if pi == 2; xlabel(ax, 'time [s]'); else; ax.XTickLabel = []; end
        style_axis(ax);
        legend(ax, handles, legendText, 'Location','best', ...
            'FontName','Times New Roman', 'FontSize',8, 'Box','on');
    end
    sgtitle(fPosition, specs{ci,2}, 'FontName','Times New Roman', ...
        'FontWeight','bold', 'FontSize',11);
    try, cd_export(fPosition, ['A2_position_' specs{ci,1}], export_pdf); catch, end
end
end

function handles = plot_panel(ax, t, S, spec, j, modelCol, policyCol)
h2 = plot(ax, t, getcol(S,spec{2},j), '-', 'Color', modelCol, 'LineWidth',1.3);
h3 = plot(ax, t, getcol(S,spec{3},j), '-', 'Color', policyCol, 'LineWidth',1.8);
h1 = plot(ax, t, getcol(S,spec{1},j), 'k-', 'LineWidth',2.4);
handles = [h1 h2 h3];
end

function style_axis(ax)
set(ax, 'FontName','Times New Roman', 'FontSize',9, ...
    'LineWidth',0.8, 'TickDir','in', 'Layer','top', ...
    'XGrid','on', 'YGrid','on', 'GridAlpha',0.18, ...
    'MinorGridAlpha',0.10);
end

function [t, j] = deal_env(S, target)
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
t = (0:T-1)*dt;
end
function y = getcol(S, field, j); A = orient(S.(field)); j=min(j,size(A,2)); y=A(:,j); end
function A = orient(A); A = double(A); if size(A,1)<size(A,2); A=A.'; end; end
