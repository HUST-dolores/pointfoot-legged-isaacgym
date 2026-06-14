function C5b_push_timetrace()
% C5b 横向推时间轨迹:选一重载 env(≈26kg),40N 持续横向力下 roll vs 时间,四变体叠加。
% 观点:扰动抑制动态——好的(Model)压住小幅,差的(RL)大幅持续晃。一张图。不自动保存。

here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
variants = {'Model-guided', 'Estimate-guided', 'Source-guided', 'RL-only'};
col = ch5_colors();
target_load = 26;

figure('Color','w','Position',[80 80 760 480],'Name','C5b 横向推轨迹'); hold on; grid on; box on;
h = gobjects(1,numel(variants)); lab = cell(1,numel(variants));
for vi = 1:numel(variants)
    [tt, rr, L] = trace_one(ED, variants{vi}, target_load);
    if isempty(tt); h(vi)=plot(nan,nan,'-','Color',col{vi}); lab{vi}=variants{vi}; continue; end
    h(vi) = plot(tt, rr, '-', 'Color', col{vi}, 'LineWidth', 1.6);
    lab{vi} = variants{vi};
end
xlabel('Time [s]'); ylabel('Roll [deg]');
legend(h, lab, 'Location','northwest');
title({'C5b Lateral push (40 N) roll response (fixed load = 26 kg)', 'sustained +Y 40 N from t=0; 26 kg load appears at t=0.5 s'}, 'FontWeight','bold');
end

function [tt, rr, L] = trace_one(ED, vname, target)
tt=[]; rr=[]; L=nan; best=''; bt=-1;
for f = dir(fullfile(ED, 'play_data_*_load26-26_flat_fxyz0-40-0N.mat'))'
    S = load(fullfile(f.folder,f.name));
    if ~isfield(S,'meta')||~contains(char(getf(S.meta,'load_run','')),'wide2-30'); continue; end
    if ~strcmp(variant_name(S.meta), vname); continue; end
    rr = orient(S.payload_mass_ref_all); if mean(any(rr>0.3,1)) < 0.5; continue; end  % 需真实负载(--keep_load)
    if f.datenum>bt; bt=f.datenum; best=fullfile(f.folder,f.name); end
end
if isempty(best); return; end
S=load(best); roll=orient(S.base_roll_all)*180/pi; ref=orient(S.payload_mass_ref_all);
dt=0.02; if isfield(S,'dt')&&~isempty(S.dt); dt=double(S.dt(1)); end
[T,N]=size(roll);
loads=arrayfun(@(j) median(ref(ref(:,j)>0.3,j),'omitnan'), 1:N);
[~,j]=min(abs(loads-target)); L=loads(j);
tt=(0:T-1)*dt; rr=roll(:,j);
end
function A = orient(A); A = double(A); if size(A,1)<size(A,2); A=A.'; end; end
function vn = variant_name(meta)
qs=logical(getf(meta,'use_qs_in_obs',true)); re=logical(getf(meta,'use_load_residual_estimation',false)); tq=logical(getf(meta,'use_torques_in_obs',true));
if qs&&re; vn='Model-guided'; elseif ~qs&&~re&&~tq; vn='RL-only'; elseif qs&&~re; vn='Estimate-guided'; else; vn='Source-guided'; end
end
function v = getf(s,f,d); if isfield(s,f)&&~isempty(s.(f)); v=s.(f); else; v=d; end; end
