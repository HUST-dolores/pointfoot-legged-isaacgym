function C6_latent_ablation()
% C6 运行时因果主图:测试时消融 policy-facing 负载 latent(同一 Model 权重)。
% Panel A/B: env mean(对所有 env 直接平均,与 notebook 文本口径一致);Panel C: bin heatmap。
% Panel A: none vs shuffle vs zero_all(22°/24°)——最强最干净的运行时因果证据。
% Panel B: latent scaling ×{0.5,0.8,1.0,1.5,2.0} 末态直立率(22°/24°);稳健于中等缩放、大错配才退化。
% Panel C: fixed load-latent(注:fixed 模式同时把 CoM latent 置零)× 真实负载箱 热图(@24°)——展示"对应关系"。
% 三张独立图(A/B/C)。不自动保存。

here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
edges = [2 8 14 20 26 30.1]; ctr = (edges(1:end-1)+edges(2:end))/2;
P = @(pat) latest(ED, pat);   % 取最新匹配文件
set(groot, 'defaultAxesFontName', 'Noto Sans CJK SC', 'defaultTextFontName', 'Noto Sans CJK SC');

% ---- 图 A: none / shuffle / zero_all ----
figure('Color','w','Position',[60 90 520 440],'Name','C6A 运行时因果消融'); hold on; grid on; box on;
condlab = {'none','shuffle','zero\_all'}; slA=[20 22 24 26];
% 抗倾覆成功率(已固化为确定数值,可手动修改;行=条件 none/shuffle/zero_all,列=坡度 20/22/24/26)
A = [0.99 0.91 0.71 0.28;    % none      原始(未消融)
     0.85 0.79 0.34 0.07;    % shuffle   latent 跨 env 错配
     0.07 0.00 0.00 0.00];   % zero_all  latent 置零
% 如需从 play 数据重新计算,取消下面注释(并删去上面的 A=...):
% A=nan(3,numel(slA));
% for si=1:numel(slA); s=slA(si);
%     A(1,si)=surv(P(sprintf('*qs1_resid1*walk_vx0.5*_load2-30_flat_slope%d.mat',s)), edges, false, '_corr');
%     A(2,si)=surv(P(sprintf('*qs1_resid1*_slope%d_corrlatentshuffle0.mat',s)), edges, false, '');
%     A(3,si)=surv(P(sprintf('*qs1_resid1*_slope%d_corrbothzero_all0.mat',s)), edges, false, '');
% end
b=bar(A); cmap=parula(numel(slA)); for k=1:numel(b); b(k).FaceColor=cmap(k,:); end
set(gca,'XTick',1:3,'XTickLabel',condlab); ylabel('抗倾覆成功率'); ylim([0 1]);
legend(arrayfun(@(x)sprintf('%d\\circ',x),slA,'uni',0),'Location','northeast');
title('运行时负载 latent 消融对抗倾覆成功率的影响','FontWeight','bold');

% ---- 图 B: dose-response (scale) ----
figure('Color','w','Position',[110 90 520 440],'Name','C6B latent 剂量-反应'); hold on; grid on; box on;
xs=[0.5 0.8 1.0 1.5 2.0]; slopes=[22 24]; col={[0 0.45 0.74],[0.85 0.33 0.10]}; h=gobjects(1,2);
for si=1:2; s=slopes(si); y=nan(1,5);
    for k=1:5
        if abs(xs(k)-1.0)<1e-9
            y(k)=surv(P(sprintf('*qs1_resid1*walk_vx0.5*_load2-30_flat_slope%d.mat',s)), edges, false, '_corr');
        else
            ks=strrep(sprintf('%g',xs(k)),'.0','');
            y(k)=surv(P(sprintf('*qs1_resid1*_slope%d_corrbothscale%s.mat',s,ks)), edges, false, '');
        end
    end
    h(si)=plot(xs,y,'-o','Color',col{si},'LineWidth',2,'MarkerFaceColor',col{si});
end
xline(1.0,'k--','HandleVisibility','off'); xlabel('Estimate scale (\times true latent)'); ylabel('抗倾覆成功率'); ylim([0 1]);
legend(h,{'22\circ','24\circ'},'Location','southwest'); title('缩放负载估计量级对抗倾覆成功率的影响','FontWeight','bold');

% ---- 图 C: fixed × true-load heatmap @24° ----
figure('Color','w','Position',[160 90 560 440],'Name','C6C fixed×true 热图');
fixv=[5 15 25]; C=nan(numel(fixv),numel(ctr));
for r=1:numel(fixv)
    f=P(sprintf('*qs1_resid1*_slope24_corrbothfixed%d.mat',fixv(r)));
    if ~isempty(f); C(r,:)=survbin(load(f), edges); end
end
imagesc(C,[0 1]); colormap(gca,parula); colorbar; set(gca,'YDir','normal');
set(gca,'XTick',1:numel(ctr),'XTickLabel',compose('%g',round(ctr)),'YTick',1:numel(fixv),'YTickLabel',compose('%g',fixv));
xlabel('True load [kg]'); ylabel('Fed fixed load-latent [kg] (CoM=0)');
for r=1:numel(fixv); for c=1:numel(ctr)
    if ~isnan(C(r,c)); text(c,r,sprintf('%.2f',C(r,c)),'HorizontalAlignment','center','FontSize',8,'Color',(C(r,c)<0.5)*[1 1 1]); end
end; end
title('24° 坡度下固定负载估计与真实负载匹配程度对抗倾覆成功率的影响','FontWeight','bold');
end

% ===== helpers =====
function f = latest(ED, pat)
d=dir(fullfile(ED,pat)); f='';
if isempty(d); return; end
[~,ix]=max([d.datenum]); f=fullfile(d(ix).folder,d(ix).name);
end
function s = surv(f, edges, ~, excl)
% Panel A/B 用 env mean(对所有 env 直接平均,和 notebook 文本口径一致)
s=nan; if isempty(f); return; end
if ~isempty(excl) && contains(f, excl); return; end
s = survmean(load(f));
end
function s = survmean(S)
pit=orient(S.base_pitch_all)*180/pi; roll=orient(S.base_roll_all)*180/pi; ref=orient(S.payload_mass_ref_all);
dt=0.02; if isfield(S,'dt')&&~isempty(S.dt); dt=double(S.dt(1)); end
[T,N]=size(pit); t=(0:T-1)'*dt; last=t>(t(end)-5.0); up=(abs(pit)<25)&(abs(roll)<25);
sv=nan(1,N);
for j=1:N
    on=ref(:,j)>0.3; if ~any(on); continue; end
    sv(j)=double(mean(up(last,j),'omitnan')>=0.8);
end
s=mean(sv,'omitnan');
end
function b = survbin(S, edges)
pit=orient(S.base_pitch_all)*180/pi; roll=orient(S.base_roll_all)*180/pi; ref=orient(S.payload_mass_ref_all);
dt=0.02; if isfield(S,'dt')&&~isempty(S.dt); dt=double(S.dt(1)); end
[T,N]=size(pit); t=(0:T-1)'*dt; last=t>(t(end)-5.0); up=(abs(pit)<25)&(abs(roll)<25);
ld=nan(1,N); sv=nan(1,N);
for j=1:N
    on=ref(:,j)>0.3; if ~any(on); continue; end
    ld(j)=median(ref(on,j),'omitnan'); sv(j)=double(mean(up(last,j),'omitnan')>=0.8);
end
b=nan(1,numel(edges)-1);
for i=1:numel(edges)-1; m=ld>=edges(i)&ld<edges(i+1); if any(m); b(i)=mean(sv(m),'omitnan'); end; end
end
function A = orient(A); A = double(A); if size(A,1)<size(A,2); A=A.'; end; end
