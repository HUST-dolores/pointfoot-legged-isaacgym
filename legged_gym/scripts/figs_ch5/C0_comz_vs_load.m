function C0_comz_vs_load()
% C0 负载质心竖直高度 vs 负载质量(物理特性,与策略无关)。
% 数据:C5PUSH-cen100 居中 play 的 true_com_delta_z(每-env 末段中位),四变体合 ~400 点。
% 用途:支撑"负载越重→质心越高→越易倾覆"的物理(C 组抗倾覆的基础)。散点(按变体着色,验证与策略无关)+ 一条总拟合线。不自动保存。

here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
variants = {'Model-guided','Estimate-guided','Source-guided','RL-only'};
col = ch5_colors();

figure('Color','w','Position',[80 80 700 520],'Name','C0 CoM-z vs 负载'); hold on; grid on; box on;
allm = []; allz = []; hs = gobjects(1,numel(variants));
for vi = 1:numel(variants)
    [m, z] = collect(ED, variants{vi});
    if isempty(m); hs(vi)=scatter(nan,nan,26,col{vi},'filled'); continue; end
    hs(vi) = scatter(m, z*100, 26, col{vi}, 'filled', 'MarkerFaceAlpha',0.45, 'MarkerEdgeColor',col{vi}, 'MarkerEdgeAlpha',0.6);
    allm=[allm m]; allz=[allz z]; %#ok<AGROW>
end
% 物理模型(质量加权质心,固定几何负载,base_task.py:1460):Δz = h·m/(m0+m)
%   m0=trunk质量(URDF 9.585kg), h=负载挂点高出trunk质心(=r-c0≈23cm)。
% 线性化 1/Δz = 1/h + (m0/h)·(1/m) 求参(免 Optimization Toolbox)。
zc = allz*100; q = polyfit(1./allm, 1./zc, 1);
hh = 1/q(2); m0 = q(1)*hh;                              % h=1/截距, m0=斜率·h
pred = hh*allm./(m0+allm); R2 = 1 - sum((zc-pred).^2)/sum((zc-mean(zc)).^2);
xf = linspace(min(allm), max(allm), 200);
hl = plot(xf, hh*xf./(m0+xf), 'k-', 'LineWidth', 2.6);
xlabel('Load mass [kg]'); ylabel('Load CoM vertical offset [cm]');
legend([hl hs], [{sprintf('fit: \\Deltaz = %.1f\\cdotm/(%.1f+m)  (R^2=%.3f, n=%d)', hh, m0, R2, numel(allm))}, variants], ...
       'Location','northwest');
title('整机质心竖直偏移随负载质量的变化', 'FontWeight','bold');
fprintf('  CoM-z saturating fit: dz[cm] = %.2f*m/(%.2f+m), R2=%.4f, n=%d  (m0=trunk mass, h=load above trunk CoM)\n', hh, m0, R2, numel(allm));
end

function [m, z] = collect(ED, vname)
m=[]; z=[]; best=''; bt=-1;
for f = dir(fullfile(ED, 'play_data_*C5PUSH-cen100*_fxyz0-40-0N.mat'))'
    S = load(fullfile(f.folder,f.name));
    if ~isfield(S,'meta')||~contains(char(getf(S.meta,'load_run','')),'wide2-30'); continue; end
    if ~strcmp(variant_name(S.meta), vname); continue; end
    if f.datenum>bt; bt=f.datenum; best=fullfile(f.folder,f.name); end
end
if isempty(best); return; end
S=load(best); ref=orient(S.payload_mass_ref_all); cz=orient(S.true_com_delta_z_all);
[T,N]=size(ref);
for j=1:N
    on=ref(:,j)>0.3; if ~any(on); continue; end
    m(end+1)=median(ref(on,j),'omitnan'); z(end+1)=median(cz(on,j),'omitnan'); %#ok<AGROW>
end
end
function A = orient(A); A = double(A); if size(A,1)<size(A,2); A=A.'; end; end
function vn = variant_name(meta)
qs=logical(getf(meta,'use_qs_in_obs',true)); re=logical(getf(meta,'use_load_residual_estimation',false)); tq=logical(getf(meta,'use_torques_in_obs',true));
if qs&&re; vn='Model-guided'; elseif ~qs&&~re&&~tq; vn='RL-only'; elseif qs&&~re; vn='Estimate-guided'; else; vn='Source-guided'; end
end
function v = getf(s,f,d); if isfield(s,f)&&~isempty(s.(f)); v=s.(f); else; v=d; end; end
