function C4_eccentric()
% C4 偏心负载前倾:静止、平地、负载 2-30,末段前倾 pitch(签名°,负=前倾)vs 负载,四变体。
% 两子图:左 x=0(居中,纯竖直重载)、右 x=0.20(前偏,偏心力矩∝m·offset)。一张图。不自动保存。

here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
variants = {'Model-guided', 'Estimate-guided', 'Source-guided', 'RL-only'};
col = ch5_colors();
edges = [2 8 14 20 26 30.1]; ctr = (edges(1:end-1)+edges(2:end))/2;
offs = {'0','0.2'}; ttl = {'x=0 (centered)','x=0.20 (forward eccentric)'};

figure('Color','w','Position',[60 80 1000 440],'Name','C4 偏心负载前倾');
for oi = 1:2
    ax = subplot(1,2,oi); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    h = gobjects(1,numel(variants));
    for vi = 1:numel(variants)
        [ld, pm] = collect(ED, variants{vi}, offs{oi});
        if isempty(ld); h(vi)=plot(ax,nan,nan,'-o','Color',col{vi}); continue; end
        mb = nan(1,numel(ctr));
        for bi=1:numel(ctr); s=ld>=edges(bi)&ld<edges(bi+1); if any(s); mb(bi)=mean(pm(s),'omitnan'); end; end
        h(vi) = plot(ax, ctr, mb, '-o', 'Color', col{vi}, 'LineWidth', 2, 'MarkerFaceColor', col{vi});
    end
    xlabel(ax,'Load [kg]'); ylabel(ax,'End-segment pitch [deg] (neg = lean forward)');
    title(ax, ttl{oi}, 'FontWeight','bold'); ylim(ax,[-12 4]);
    if oi==2; legend(ax, h, variants, 'Location','southwest'); end
end
sgtitle('C4 Forward lean under load (static, flat): model-based group leans less', 'FontWeight','bold');
end

function [ld, pm] = collect(ED, vname, off)
ld=[]; pm=[]; best=''; bt=-1;
for f = dir(fullfile(ED, sprintf('play_data_*_ecc%sx0y.mat', off)))'
    S = load(fullfile(f.folder,f.name));
    if ~isfield(S,'meta')||~contains(char(getf(S.meta,'load_run','')),'wide2-30'); continue; end
    if ~strcmp(variant_name(S.meta), vname); continue; end
    if f.datenum>bt; bt=f.datenum; best=fullfile(f.folder,f.name); end
end
if isempty(best); return; end
S=load(best); pit=orient(S.base_pitch_all)*180/pi; ref=orient(S.payload_mass_ref_all);
dt=0.02; if isfield(S,'dt')&&~isempty(S.dt); dt=double(S.dt(1)); end
[T,N]=size(pit); t=(0:T-1)'*dt; last=t>(t(end)-5.0);
for j=1:N
    on=ref(:,j)>0.3; L=median(ref(on,j),'omitnan'); if isnan(L); continue; end
    ld(end+1)=L; pm(end+1)=mean(pit(last,j),'omitnan'); %#ok<AGROW>
end
end
function A = orient(A); A = double(A); if size(A,1)<size(A,2); A=A.'; end; end
function vn = variant_name(meta)
qs=logical(getf(meta,'use_qs_in_obs',true)); re=logical(getf(meta,'use_load_residual_estimation',false)); tq=logical(getf(meta,'use_torques_in_obs',true));
if qs&&re; vn='Model-guided'; elseif ~qs&&~re&&~tq; vn='RL-only'; elseif qs&&~re; vn='Estimate-guided'; else; vn='Source-guided'; end
end
function v = getf(s,f,d); if isfield(s,f)&&~isempty(s.(f)); v=s.(f); else; v=d; end; end
