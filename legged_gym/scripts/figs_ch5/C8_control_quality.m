function C8_control_quality()
% C8 负载估计对“控制质量”的影响(仅取存活 env,排除摔倒,故与 C1 存活热图互补不重复)。
% 同一 Model-guided 权重,full(有负载 latent) vs ablate(corrbothzero_all,抹去负载 latent)。
% 坡度带 {15,16,17}° × 500 env,对坡度平均;按真实负载分箱。
% (a) 速度跟踪误差 |v_x - cmd_x|(末5s均值):主指标,无估计明显更差。
% (b) 机体高度(末5s均值):有估计主动蹲低、空载名义高度更高。一图两子图。不自动保存。

here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
set(groot, 'defaultAxesFontName', 'Noto Sans CJK SC', 'defaultTextFontName', 'Noto Sans CJK SC');
edges = [2 8 14 20 26 30.1]; ctr = (edges(1:end-1)+edges(2:end))/2;
slopes = [15 16 17];
P = @(pat) latest(ED, pat);

TKf=nan(numel(slopes),numel(ctr)); TKa=TKf; HGf=TKf; HGa=TKf;
for si=1:numel(slopes)
    s=slopes(si);
    F=qual(load(P(sprintf('play_data_*qs1_resid1_walk_vx0.5*_load2-30_flat_slope%d.mat',s))), edges);
    A=qual(load(P(sprintf('play_data_*qs1_resid1*_load2-30_flat_slope%d_corrbothzero_all0.mat',s))), edges);
    TKf(si,:)=F.trk; TKa(si,:)=A.trk; HGf(si,:)=F.hgt; HGa(si,:)=A.hgt;
end
trk_f=mean(TKf,1,'omitnan'); trk_a=mean(TKa,1,'omitnan');
hgt_f=mean(HGf,1,'omitnan'); hgt_a=mean(HGa,1,'omitnan');

cF=[0.20 0.30 0.70]; cA=[0.85 0.33 0.10];
figure('Color','w','Position',[60 90 940 400],'Name','C8 控制质量');

subplot(1,2,1); hold on; grid on; box on;
h1=plot(ctr,trk_f,'-o','Color',cF,'LineWidth',2.2,'MarkerFaceColor',cF,'MarkerSize',7);
h2=plot(ctr,trk_a,'-s','Color',cA,'LineWidth',2.2,'MarkerFaceColor',cA,'MarkerSize',7);
xlabel('负载质量 [kg]'); ylabel('速度跟踪误差 |v_x - cmd| [m/s]'); ylim([0 1]);
legend([h1 h2],{sprintf('有估计 (full)  均值 %.2f',mean(trk_f,'omitnan')), ...
                sprintf('无估计 (ablate) 均值 %.2f',mean(trk_a,'omitnan'))},'Location','northwest');
title('(a) 承载爬坡的速度跟踪质量','FontWeight','bold');

subplot(1,2,2); hold on; grid on; box on;
h1=plot(ctr,hgt_f,'-o','Color',cF,'LineWidth',2.2,'MarkerFaceColor',cF,'MarkerSize',7);
h2=plot(ctr,hgt_a,'-s','Color',cA,'LineWidth',2.2,'MarkerFaceColor',cA,'MarkerSize',7);
xlabel('负载质量 [kg]'); ylabel('机体高度 [m]');
legend([h1 h2],{'有估计 (full)','无估计 (ablate)'},'Location','best');
title('(b) 承载下的机体高度(站姿)','FontWeight','bold');

sgtitle('负载估计对承载运动控制质量的影响(仅存活 env)','FontWeight','bold','FontName','Noto Sans CJK SC');

fprintf('--- C8 (band 15-17, survivors) ---\n');
fprintf('load   :'); fprintf(' %6.1f',ctr); fprintf('\n');
fprintf('trk f  :'); fprintf(' %6.3f',trk_f); fprintf('\n');
fprintf('trk a  :'); fprintf(' %6.3f',trk_a); fprintf('\n');
fprintf('hgt f  :'); fprintf(' %6.3f',hgt_f); fprintf('\n');
fprintf('hgt a  :'); fprintf(' %6.3f',hgt_a); fprintf('\n');
end

% ===== helpers =====
function f = latest(ED, pat)
d=dir(fullfile(ED,pat)); f='';
d=d(~contains({d.name},'estop')&~contains({d.name},'static'));
if isempty(d); return; end; [~,ix]=max([d.datenum]); f=fullfile(d(ix).folder,d(ix).name);
end
function q = qual(S, edges)
pit=orient(S.base_pitch_all)*180/pi; roll=orient(S.base_roll_all)*180/pi;
h=orient(S.base_height_all); vx=orient(S.base_lin_vel_x_all); cmd=orient(S.command_x_all); ref=orient(S.payload_mass_ref_all);
dt=0.02; if isfield(S,'dt')&&~isempty(S.dt); dt=double(S.dt(1)); end
[T,N]=size(pit); t=(0:T-1)'*dt; last=t>(t(end)-5.0); up=(abs(pit)<25)&(abs(roll)<25);
ld=nan(1,N); trk=nan(1,N); hgt=nan(1,N); sv=nan(1,N);
for j=1:N
    on=ref(:,j)>0.3; if ~any(on); continue; end
    ld(j)=median(ref(on,j),'omitnan');
    trk(j)=mean(abs(vx(last,j)-cmd(last,j)),'omitnan');
    hgt(j)=mean(h(last,j),'omitnan');
    sv(j)=double(mean(up(last,j),'omitnan')>=0.8);
end
keep=sv>0.5;
q.trk=binm(ld,trk,keep,edges); q.hgt=binm(ld,hgt,keep,edges);
end
function b = binm(ld,v,keep,edges)
b=nan(1,numel(edges)-1);
for i=1:numel(edges)-1
    m=ld>=edges(i)&ld<edges(i+1)&keep; if any(m); b(i)=mean(v(m),'omitnan'); end
end
end
function A = orient(A); A = double(A); if size(A,1)<size(A,2); A=A.'; end; end
