function C7_value_of_estimation()
% 图5.11 强化学习中负载估计的控制价值 V(m_L) 随负载的变化。
% 定义(eq5.10 的连续抗倾覆代价版本):J 取斜坡上稳态机体倾角(|pitch| 末5s 均值,越小越稳),
%   V(m_L)=J_ablate(m_L)-J_full(m_L) = 抹去 policy-facing 负载 latent(corrbothzero_all,
%   仅置零 mass/CoM,保留速度)后多付出的倾角代价,按真实负载分箱。
% 这是“成功率/β* 会饱和→对负载不敏感”的连续替代量,与推论5.1 解析增益 Δτ_res∝m_L 同为连续量、同向。
% 数据:Model-guided seed1 ckpt11000,坡度带 {15,16,17}°(消融呈负载分级的区间)×500 env,
%   对坡度求平均。叠加线性拟合(斜率>0,示意与解析 Δτ_res∝m_L 同向)。一张图。不自动保存。

here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
set(groot, 'defaultAxesFontName', 'Noto Sans CJK SC', 'defaultTextFontName', 'Noto Sans CJK SC');

edges = [2 8 14 20 26 30.1]; ctr = (edges(1:end-1)+edges(2:end))/2;
slopes = [15 16 17];
P = @(pat) latest(ED, pat);

Vs = nan(numel(slopes), numel(ctr));   % 每坡度的 V(额外倾角)
for si = 1:numel(slopes)
    s = slopes(si);
    ff = P(sprintf('play_data_*qs1_resid1_walk_vx0.5*_load2-30_flat_slope%d.mat', s));
    fa = P(sprintf('play_data_*qs1_resid1*_load2-30_flat_slope%d_corrbothzero_all0.mat', s));
    Jf = leanbin(load(ff), edges);
    Ja = leanbin(load(fa), edges);
    Vs(si,:) = Ja - Jf;
end
V = mean(Vs, 1, 'omitnan');            % 对坡度带平均

% 线性拟合(示意与解析 Δτ_res∝m_L 同向)
ok = ~isnan(V); p = polyfit(ctr(ok), V(ok), 1);
mg = linspace(min(ctr)-1, max(ctr)+1, 100); fit = polyval(p, mg);

fprintf('--- C7 V(m_L) 额外稳态倾角 [deg] ---\n');
fprintf('load bin :'); fprintf(' %6.1f', ctr); fprintf('\n');
fprintf('V        :'); fprintf(' %6.2f', V); fprintf('\n');
fprintf('linear fit: slope=%.3f deg/kg, intercept=%.2f deg\n', p(1), p(2));

figure('Color','w','Position',[80 90 660 470],'Name','C7 负载估计的控制价值'); hold on; grid on; box on;
h2 = plot(mg, fit, '--', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.8);
h1 = plot(ctr, V, '-o', 'Color', [0.20 0.30 0.70], 'LineWidth', 2.4, ...
          'MarkerFaceColor', [0.20 0.30 0.70], 'MarkerSize', 8);
xlabel('负载质量 [kg]');
ylabel('控制价值 V(m_L) = J_{ablate} - J_{full}  [\circ]');
xlim([min(ctr)-1.5 max(ctr)+1.5]); ylim([0 max(8, max(V(ok))*1.2)]);
legend([h1 h2], {'V(m_L)(消融多付的稳态倾角)', '线性拟合 \propto m_L(与解析 \Delta\tau_{res} 同向)'}, ...
       'Location','northwest');
title('强化学习中负载估计的控制价值随负载的变化', 'FontWeight','bold');
end

% ===== helpers =====
function f = latest(ED, pat)
d = dir(fullfile(ED, pat)); f = '';
d = d(~contains({d.name},'estop') & ~contains({d.name},'static'));
if isempty(d); return; end
[~,ix] = max([d.datenum]); f = fullfile(d(ix).folder, d(ix).name);
end
function b = leanbin(S, edges)
% 每 env:载荷期内 |pitch| 末5s 均值(稳态机体倾角,deg);按真实负载分箱取均值。
pit = orient(S.base_pitch_all)*180/pi; ref = orient(S.payload_mass_ref_all);
dt = 0.02; if isfield(S,'dt')&&~isempty(S.dt); dt = double(S.dt(1)); end
[T,N] = size(pit); t = (0:T-1)'*dt; last = t > (t(end)-5.0);
ld = nan(1,N); ln = nan(1,N);
for j = 1:N
    on = ref(:,j) > 0.3; if ~any(on); continue; end
    ld(j) = median(ref(on,j),'omitnan');
    ln(j) = mean(abs(pit(last,j)),'omitnan');
end
b = nan(1,numel(edges)-1);
for i = 1:numel(edges)-1
    m = ld>=edges(i) & ld<edges(i+1); if any(m); b(i) = mean(ln(m),'omitnan'); end
end
end
function A = orient(A); A = double(A); if size(A,1)<size(A,2); A=A.'; end; end
