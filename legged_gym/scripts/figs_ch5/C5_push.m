function C5_push()
% C5 横向推 40N(真载下)抗扰:静止、平地、负载 2-30、+Y 持续 40N,末段峰值 |roll| vs 负载,四变体。
% 负载位置居中(--load_offset 0,0,100 env),只保留质量随机,降低随机性。
% 数据:exported/*C5PUSH-cen100*_fxyz0-40-0N.mat(无则回退旧随机位置数据)。一张图(4 曲线)。不自动保存。

here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
variants = {'Model-guided', 'Estimate-guided', 'Source-guided', 'RL-only'};
col = ch5_colors();
edges = [2 8 14 20 26 30.1]; ctr = (edges(1:end-1)+edges(2:end))/2;

figure('Color','w','Position',[80 80 720 480],'Name','C5 横向推 40N'); hold on; grid on; box on;
h = gobjects(1,numel(variants)); lab = cell(1,numel(variants));
for vi = 1:numel(variants)
    [ld, rl] = collect(ED, variants{vi});
    if isempty(ld); h(vi)=plot(nan,nan,'-o','Color',col{vi}); lab{vi}=variants{vi}; continue; end
    scatter(ld, rl, 22, col{vi}, 'filled', 'MarkerFaceAlpha', 0.55, 'MarkerEdgeColor', col{vi}, 'MarkerEdgeAlpha', 0.8, 'HandleVisibility','off');
    mb = nan(1,numel(ctr));
    for bi=1:numel(ctr); s=ld>=edges(bi)&ld<edges(bi+1); if any(s); mb(bi)=mean(rl(s),'omitnan'); end; end
    h(vi) = plot(ctr, mb, '-o', 'Color', col{vi}, 'LineWidth', 2, 'MarkerFaceColor', col{vi});
    mu = mean(rl,'omitnan');                                   % 总体均值峰值|roll|
    lab{vi} = sprintf('%s  (mean %.1f\\circ)', variants{vi}, mu);
    fprintf('  %-16s mean peak|roll| = %.1f deg (n=%d)\n', variants{vi}, mu, numel(rl));
end
xlabel('Load [kg]'); ylabel('Peak |roll| [deg]'); ylim([0 20]);
legend(h, lab, 'Location','northwest');
title({'C5 Lateral push (40 N) disturbance rejection vs load', 'static, flat, centered load 2-30kg (100 envs)'}, 'FontWeight','bold');
end

function [ld, rl] = collect(ED, vname)
ld=[]; rl=[]; best=''; bt=-1;
pat = 'play_data_*C5PUSH-cen100*_fxyz0-40-0N.mat';                       % 居中专用(优先)
if isempty(dir(fullfile(ED,pat))); pat='play_data_*_load2-30_flat_fxyz0-40-0N.mat'; end  % 回退旧随机
for f = dir(fullfile(ED, pat))'
    S = load(fullfile(f.folder,f.name));
    if ~isfield(S,'meta')||~contains(char(getf(S.meta,'load_run','')),'wide2-30'); continue; end
    if ~strcmp(variant_name(S.meta), vname); continue; end
    rr = orient(S.payload_mass_ref_all); if mean(any(rr>0.3,1)) < 0.5; continue; end  % 需真实负载(--keep_load),排除无载力实验
    if f.datenum>bt; bt=f.datenum; best=fullfile(f.folder,f.name); end
end
if isempty(best); return; end
S=load(best); roll=orient(S.base_roll_all)*180/pi; ref=orient(S.payload_mass_ref_all);
dt=0.02; if isfield(S,'dt')&&~isempty(S.dt); dt=double(S.dt(1)); end
[T,N]=size(roll); t=(0:T-1)'*dt; last=t>(t(end)-5.0);
for j=1:N
    on=ref(:,j)>0.3; L=median(ref(on,j),'omitnan'); if isnan(L); continue; end
    ld(end+1)=L; rl(end+1)=max(abs(roll(last,j))); %#ok<AGROW>
end
end
function A = orient(A); A = double(A); if size(A,1)<size(A,2); A=A.'; end; end
function vn = variant_name(meta)
qs=logical(getf(meta,'use_qs_in_obs',true)); re=logical(getf(meta,'use_load_residual_estimation',false)); tq=logical(getf(meta,'use_torques_in_obs',true));
if qs&&re; vn='Model-guided'; elseif ~qs&&~re&&~tq; vn='RL-only'; elseif qs&&~re; vn='Estimate-guided'; else; vn='Source-guided'; end
end
function v = getf(s,f,d); if isfield(s,f)&&~isempty(s.(f)); v=s.(f); else; v=d; end; end
