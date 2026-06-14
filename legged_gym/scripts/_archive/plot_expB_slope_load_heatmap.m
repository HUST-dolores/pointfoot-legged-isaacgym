function plot_expB_slope_load_heatmap()
% 实验B/Ch5预备 — 每个策略一张"负载 × 坡度 → 存活率"热力图。
% 读 exported/ 下 *_load2-30_flat_slope*.mat(--load_hold --slope_deg 的 2D 扫描)。
% 存活率 = t>1.5s 内 |pitch|<25° & |roll|<25° 的时间占比，按负载分箱、跨同箱 env 取均值。

here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
variants = {'Model-guided', 'Source-guided', 'Estimate-guided', 'RL-only'};
loadEdges = [2 6 10 14 18 22 26 30];                 % 负载分箱边
loadCtr = (loadEdges(1:end-1) + loadEdges(2:end)) / 2;
slopes = [0 8 16 20 24 28];

% 收集每个 (variant, slope) 文件的 per-env (load, survival)，取最新。
% 仅取行走斜坡存活，排除 static/estop/corrupt 消融文件。
files = dir(fullfile(ED, 'play_data_*walk_vx0.5*_load2-30_flat_slope*.mat'));
data = containers.Map();                              % key "vi_slope" -> [load; surv]
mtime = containers.Map();
for k = 1:numel(files)
    fn = files(k).name; fp = fullfile(files(k).folder, fn);
    if contains(fn, 'estop') || contains(fn, 'static') || contains(fn, '_corr'); continue; end
    sl = regexp(fn, 'slope([0-9.]+)', 'tokens', 'once'); if isempty(sl); continue; end
    sl = str2double(sl{1});
    S = load(fp); if ~isfield(S, 'meta'); continue; end
    vi = find(strcmp(variants, variant_name(S.meta)), 1); if isempty(vi); continue; end
    key = sprintf('%d_%g', vi, sl);
    if isKey(mtime, key) && mtime(key) >= files(k).datenum; continue; end
    [ld, sv] = env_survival(S);
    data(key) = [ld(:)'; sv(:)']; mtime(key) = files(k).datenum;
end

fig = figure('Color', 'w', 'Position', [40 40 1100 760], 'Name', 'Exp B 负载×坡度 存活率');
for vi = 1:4
    M = nan(numel(slopes), numel(loadCtr));
    for si = 1:numel(slopes)
        key = sprintf('%d_%g', vi, slopes(si));
        if ~isKey(data, key); continue; end
        d = data(key); ld = d(1, :); sv = d(2, :);
        for bi = 1:numel(loadCtr)
            sel = ld >= loadEdges(bi) & ld < loadEdges(bi+1);
            if any(sel); M(si, bi) = mean(sv(sel), 'omitnan'); end
        end
    end
    ax = subplot(2, 2, vi);
    imagesc(ax, loadCtr, slopes, M, [0 1]); set(ax, 'YDir', 'normal'); colormap(ax, parula);
    xlabel(ax, '负载 [kg]'); ylabel(ax, '坡度 [°]'); title(ax, variants{vi}, 'FontWeight', 'bold');
    colorbar(ax);
    % 叠数字
    for si = 1:numel(slopes)
        for bi = 1:numel(loadCtr)
            if ~isnan(M(si, bi))
                text(ax, loadCtr(bi), slopes(si), sprintf('%.2f', M(si, bi)), ...
                    'HorizontalAlignment', 'center', 'FontSize', 7, ...
                    'Color', (M(si, bi) < 0.5) * [1 1 1]);
            end
        end
    end
end
sgtitle('负载 × 坡度 → 存活率(t>1.5s 直立占比)。颜色越暖越稳；RL-only 在坡上大负载明显塌', 'FontWeight', 'bold');
try
    exportgraphics(fig, fullfile(ED, 'expB_slope_load_heatmap.pdf'), 'ContentType', 'vector');
    exportgraphics(fig, fullfile(ED, 'expB_slope_load_heatmap.png'), 'Resolution', 150);
catch; saveas(fig, fullfile(ED, 'expB_slope_load_heatmap.png')); end
fprintf('[expB] saved heatmap to %s\n', ED);
end

% ------------------------------------------------------------------
function [ld, sv] = env_survival(S)
enc = orient(S.base_pitch_all) * 180 / pi; roll = orient(S.base_roll_all) * 180 / pi;
ref = orient(S.payload_mass_ref_all);
dt = 0.02; if isfield(S, 'dt') && ~isempty(S.dt); dt = double(S.dt(1)); end
[T, N] = size(enc); t = (0:T-1)' * dt; after = t > 1.5;
up = (abs(enc) < 25) & (abs(roll) < 25);
ld = zeros(1, N); sv = zeros(1, N);
for j = 1:N
    on = ref(:, j) > 0.3;
    ld(j) = median(ref(on, j), 'omitnan'); if isnan(ld(j)); ld(j) = 0; end
    sv(j) = mean(up(after, j), 'omitnan');
end
end

function A = orient(A)
A = double(A); if size(A, 1) < size(A, 2); A = A.'; end
end

function vn = variant_name(meta)
gf = @(f, d) ternary(isfield(meta, f) && ~isempty(meta.(f)), @() meta.(f), d);
qs = logical(gf('use_qs_in_obs', true));
re = logical(gf('use_load_residual_estimation', false));
tq = logical(gf('use_torques_in_obs', true));
if qs && re; vn = 'Model-guided';
elseif qs && ~re; vn = 'Estimate-guided';
elseif ~qs && ~re && tq; vn = 'Source-guided';
else; vn = 'RL-only'; end
end

function r = ternary(c, a, b)
if c; r = a(); else; r = b; end
end
