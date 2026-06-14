function C3b_estop_timetrace()
% C3b 下坡急停时间轨迹:选一重载 env(≈26kg),从指令归零起 机体x位移 vs 时间,四变体叠加。
% 观点:"怎么停的"——好的(Model)平稳停住,差的(RL)冲过更远。一张图。不自动保存。

here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
variants = {'Model-guided', 'Estimate-guided', 'Source-guided', 'RL-only'};
col = ch5_colors();
target_load = 26;

figure('Color','w','Position',[80 80 760 480],'Name','C3b 急停轨迹'); hold on; grid on; box on;
h = gobjects(1,numel(variants)); lab = cell(1,numel(variants));
for vi = 1:numel(variants)
    [tt, dd, L] = trace_one(ED, variants{vi}, target_load);
    if isempty(tt); h(vi)=plot(nan,nan,'-','Color',col{vi}); lab{vi}=variants{vi}; continue; end
    h(vi) = plot(tt, dd, '-', 'Color', col{vi}, 'LineWidth', 2);
    lab{vi} = variants{vi};
end
xline(0,'k--','HandleVisibility','off');
xlabel('Time since command\rightarrow0 [s]'); ylabel('Forward displacement [m]');
legend(h, lab, 'Location','northwest');
title('搭载 26 kg 负载下坡 10° 急停后机体的刹车记录', 'FontWeight','bold');
end

function [tt, dd, L] = trace_one(ED, vname, target)
tt=[]; dd=[]; L=nan; best=''; bt=-1;
for f = dir(fullfile(ED, 'play_data_*_load26-26_flat_slope10_estop1.5.mat'))'
    S = load(fullfile(f.folder,f.name));
    if ~isfield(S,'meta')||~contains(char(getf(S.meta,'load_run','')),'wide2-30'); continue; end
    if ~strcmp(variant_name(S.meta), vname); continue; end
    if f.datenum>bt; bt=f.datenum; best=fullfile(f.folder,f.name); end
end
if isempty(best); return; end
S=load(best); cmd=orient(S.command_x_all); px=orient(S.base_pos_x_all); ref=orient(S.payload_mass_ref_all);
dt=0.02; if isfield(S,'dt')&&~isempty(S.dt); dt=double(S.dt(1)); end
[T,N]=size(cmd);
% 选负载最接近 target 的 env
loads=arrayfun(@(j) median(ref(ref(:,j)>0.3,j),'omitnan'), 1:N);
[~,j]=min(abs(loads-target)); L=loads(j);
pp=round(0.3/dt); hold_=round(0.5/dt); pre=round(1.0/dt); win=round(3.0/dt);
for k=(pp+1):(T-win)
    if all(cmd(k-pp:k-1,j)>1.0) && all(cmd(k:k+hold_-1,j)<0.5)
        a=max(1,k-pre); b=min(T,k+win);
        tt=((a:b)-k)*dt; dd=px(a:b,j)-px(k,j); return;
    end
end
end
function A = orient(A); A = double(A); if size(A,1)<size(A,2); A=A.'; end; end
function vn = variant_name(meta)
qs=logical(getf(meta,'use_qs_in_obs',true)); re=logical(getf(meta,'use_load_residual_estimation',false)); tq=logical(getf(meta,'use_torques_in_obs',true));
if qs&&re; vn='Model-guided'; elseif ~qs&&~re&&~tq; vn='RL-only'; elseif qs&&~re; vn='Estimate-guided'; else; vn='Source-guided'; end
end
function v = getf(s,f,d); if isfield(s,f)&&~isempty(s.(f)); v=s.(f); else; v=d; end; end
