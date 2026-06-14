function A1_estimation_accuracy()
% A1 宽负载(2-30kg)质量估计精度:encoder 末段估计 vs 真值(逐 env),四变体散点 + 单位线 + RMSE。
% 桥接第四章(方法)→第五章(大负载):证明估计方法在 2-30kg 仍精确。一张图。不自动保存。

here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
variants = {'Model-guided', 'Estimate-guided', 'Source-guided', 'RL-only'};
% 约定配色(payload_method_color / Okabe-Ito):Model=E1橙 Estimate=E2绿 Source=E3天蓝 RL-only=E4朱红
col = ch5_colors();

figure('Color','w','Position',[80 80 600 560],'Name','A1 估计精度'); hold on; grid on; box on;
plot([0 32],[0 32],'k--','LineWidth',1);   % 单位线 y=x
xfit = [2 30];   % 拟合线绘制区间
h = gobjects(1,numel(variants)); lab = cell(1,numel(variants));
for vi = 1:numel(variants)
    [tru, est] = collect(ED, variants{vi});
    if isempty(tru); h(vi)=plot(nan,nan,'-','Color',col{vi}); lab{vi}=[variants{vi} ' (无数据)']; continue; end
    m = isfinite(tru) & isfinite(est); tru=tru(m); est=est(m);
    % 散点淡化作背景
    scatter(tru, est, 22, col{vi}, 'filled', 'MarkerFaceAlpha', 0.55, 'MarkerEdgeColor', col{vi}, 'MarkerEdgeAlpha', 0.8, 'HandleVisibility','off');
    % 线性回归:est ≈ a*true + b
    p = polyfit(tru, est, 1); a = p(1); b = p(2);
    rmse = sqrt(mean((est-tru).^2));
    h(vi) = plot(xfit, a*xfit + b, '-', 'Color', col{vi}, 'LineWidth', 2.2);  % 拟合线=图例句柄
    lab{vi} = sprintf('%s:  y=%.2fx%+.2f  (RMSE %.2f)', variants{vi}, a, b, rmse);
    fprintf('  %-16s slope=%.2f intercept=%+.2f RMSE=%.2f kg\n', variants{vi}, a, b, rmse);
end
xlabel('True load [kg]'); ylabel('Encoder estimate [kg]');
legend([plot(nan,nan,'k--') h], [{'y = x (ideal)'} lab], 'Location','northwest');
title('2–30 kg 负载范围内各方案的负载质量估计精度','FontWeight','bold');
axis([0 32 0 32]); axis square;
end

function [tru, est] = collect(ED, vname)
tru=[]; est=[]; best=''; bt=-1;
for f = dir(fullfile(ED, 'play_data_*walk_vx0.5*_load2-30_flat.mat'))'
    if contains(f.name,'slope')||contains(f.name,'estop'); continue; end
    S = load(fullfile(f.folder,f.name));
    if ~isfield(S,'meta')||~contains(char(getf(S.meta,'load_run','')),'wide2-30'); continue; end
    if ~strcmp(variant_name(S.meta), vname); continue; end
    if f.datenum>bt; bt=f.datenum; best=fullfile(f.folder,f.name); end
end
if isempty(best); return; end
S=load(best); E=orient(S.encoder_mass_all); R=orient(S.payload_mass_ref_all);
dt=0.02; if isfield(S,'dt')&&~isempty(S.dt); dt=double(S.dt(1)); end
[T,N]=size(E); t=(0:T-1)'*dt; last=t>5.0;
for j=1:N
    on=R(:,j)>0.3; if ~any(on); continue; end
    tru(end+1)=median(R(on,j),'omitnan'); est(end+1)=mean(E(last,j),'omitnan'); %#ok<AGROW>
end
end
function A = orient(A); A = double(A); if size(A,1)<size(A,2); A=A.'; end; end
function vn = variant_name(meta)
qs=logical(getf(meta,'use_qs_in_obs',true)); re=logical(getf(meta,'use_load_residual_estimation',false)); tq=logical(getf(meta,'use_torques_in_obs',true));
if qs&&re; vn='Model-guided'; elseif ~qs&&~re&&~tq; vn='RL-only'; elseif qs&&~re; vn='Estimate-guided'; else; vn='Source-guided'; end
end
function v = getf(s,f,d); if isfield(s,f)&&~isempty(s.(f)); v=s.(f); else; v=d; end; end
