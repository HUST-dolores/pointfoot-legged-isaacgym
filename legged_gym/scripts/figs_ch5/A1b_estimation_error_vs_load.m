function A1b_estimation_error_vs_load()
% A1b 质量估计误差 vs 负载:逐 env 末段 |估计-真值| 按负载分箱均值,四变体。
% 观点:全程 2-30kg 误差不随负载发散(方法可平移;旧窄[2,4]基底约 4kg 饱和)。一张图。不自动保存。

here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
variants = {'Model-guided', 'Estimate-guided', 'Source-guided', 'RL-only'};
col = ch5_colors();
edges = [2 8 14 20 26 30.1]; ctr = (edges(1:end-1)+edges(2:end))/2;

figure('Color','w','Position',[80 80 720 480],'Name','A1b 估计误差vs负载'); hold on; grid on; box on;
h = gobjects(1,numel(variants));
for vi = 1:numel(variants)
    [tru, err] = collect(ED, variants{vi});
    if isempty(tru); h(vi)=plot(nan,nan,'-o','Color',col{vi}); continue; end
    mb = nan(1,numel(ctr));
    for bi=1:numel(ctr); s=tru>=edges(bi)&tru<edges(bi+1); if any(s); mb(bi)=mean(abs(err(s)),'omitnan'); end; end
    h(vi) = plot(ctr, mb, '-o', 'Color', col{vi}, 'LineWidth', 2, 'MarkerFaceColor', col{vi});
end
xlabel('True load [kg]'); ylabel('Mean abs. estimation error [kg]');
legend(h, variants, 'Location','northwest');
title({'A1b Mass-estimation error vs load (wide 2-30kg)', 'model-based variants stay accurate across the whole range'}, 'FontWeight','bold');
end

function [tru, err] = collect(ED, vname)
tru=[]; err=[]; best=''; bt=-1;
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
    L=median(R(on,j),'omitnan'); tru(end+1)=L; err(end+1)=mean(E(last,j),'omitnan')-L; %#ok<AGROW>
end
end
function A = orient(A); A = double(A); if size(A,1)<size(A,2); A=A.'; end; end
function vn = variant_name(meta)
qs=logical(getf(meta,'use_qs_in_obs',true)); re=logical(getf(meta,'use_load_residual_estimation',false)); tq=logical(getf(meta,'use_torques_in_obs',true));
if qs&&re; vn='Model-guided'; elseif ~qs&&~re&&~tq; vn='RL-only'; elseif qs&&~re; vn='Estimate-guided'; else; vn='Source-guided'; end
end
function v = getf(s,f,d); if isfield(s,f)&&~isempty(s.(f)); v=s.(f); else; v=d; end; end
