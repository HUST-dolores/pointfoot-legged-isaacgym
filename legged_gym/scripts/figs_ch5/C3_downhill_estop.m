function C3_downhill_estop()
% C3 下坡 10° 急停停车距离 vs 负载,四变体。巡航 1.5→0,停车距离=指令归零起机体 x 最大前移。
% 数据:exported/*_load2-30_flat_slope10_estop1.5.mat。一张图(4 曲线)。不自动保存。

here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
variants = {'Model-guided', 'Estimate-guided', 'Source-guided', 'RL-only'};
col = ch5_colors();
edges = [2 8 14 20 26 30.1]; ctr = (edges(1:end-1)+edges(2:end))/2;

figure('Color','w','Position',[80 80 720 480],'Name','C3 下坡急停'); hold on; grid on; box on;
h = gobjects(1,numel(variants)); lab = cell(1,numel(variants));
for vi = 1:numel(variants)
    [ld, ds] = collect(ED, variants{vi});
    if isempty(ld); h(vi)=plot(nan,nan,'-o','Color',col{vi}); lab{vi}=variants{vi}; continue; end
    scatter(ld, ds, 22, col{vi}, 'filled', 'MarkerFaceAlpha', 0.55, 'MarkerEdgeColor', col{vi}, 'MarkerEdgeAlpha', 0.8, 'HandleVisibility','off');
    mb = nan(1,numel(ctr));
    for bi=1:numel(ctr); s=ld>=edges(bi)&ld<edges(bi+1); if any(s); mb(bi)=mean(ds(s),'omitnan'); end; end
    h(vi) = plot(ctr, mb, '-o', 'Color', col{vi}, 'LineWidth', 2, 'MarkerFaceColor', col{vi});
    mu = mean(ds,'omitnan');                                  % 总体均值停车距离
    lab{vi} = sprintf('%s  (mean %.2f m)', variants{vi}, mu);
    fprintf('  %-16s mean stop dist = %.2f m (n=%d)\n', variants{vi}, mu, numel(ds));
end
xlabel('Load [kg]'); ylabel('Stopping distance [m]');
legend(h, lab, 'Location','northwest');
title({'C3 Downhill (10\circ) e-stop distance vs load', 'cruise 1.5 m/s \rightarrow 0'}, 'FontWeight','bold');
end

function [ld, ds] = collect(ED, vname)
ld=[]; ds=[]; best=''; bt=-1;
for f = dir(fullfile(ED, 'play_data_*_load2-30_flat_slope10_estop1.5.mat'))'
    S = load(fullfile(f.folder,f.name));
    if ~isfield(S,'meta')||~contains(char(getf(S.meta,'load_run','')),'wide2-30'); continue; end
    if ~strcmp(variant_name(S.meta), vname); continue; end
    if f.datenum>bt; bt=f.datenum; best=fullfile(f.folder,f.name); end
end
if isempty(best); return; end
S=load(best); cmd=orient(S.command_x_all); px=orient(S.base_pos_x_all); ref=orient(S.payload_mass_ref_all);
dt=0.02; if isfield(S,'dt')&&~isempty(S.dt); dt=double(S.dt(1)); end
[T,N]=size(cmd); pp=round(0.3/dt); hold_=round(0.5/dt); win=round(2.5/dt);
for j=1:N
    on=ref(:,j)>0.3; L=median(ref(on,j),'omitnan'); if isnan(L); continue; end
    for k=(pp+1):(T-win)
        if all(cmd(k-pp:k-1,j)>1.0) && all(cmd(k:k+hold_-1,j)<0.5)
            ld(end+1)=L; ds(end+1)=max(px(k:k+win-1,j)-px(k,j)); break; %#ok<AGROW>
        end
    end
end
end
function A = orient(A); A = double(A); if size(A,1)<size(A,2); A=A.'; end; end
function vn = variant_name(meta)
qs=logical(getf(meta,'use_qs_in_obs',true)); re=logical(getf(meta,'use_load_residual_estimation',false)); tq=logical(getf(meta,'use_torques_in_obs',true));
if qs&&re; vn='Model-guided'; elseif ~qs&&~re&&~tq; vn='RL-only'; elseif qs&&~re; vn='Estimate-guided'; else; vn='Source-guided'; end
end
function v = getf(s,f,d); if isfield(s,f)&&~isempty(s.(f)); v=s.(f); else; v=d; end; end
