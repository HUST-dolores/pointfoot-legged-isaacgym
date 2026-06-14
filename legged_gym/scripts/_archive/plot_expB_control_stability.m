function plot_expB_control_stability()
% 实验B — 平地、3kg 居中负载下各变体的控制稳定性，及负载引起的退化(3kg − 0kg)。
%
%   读取 exported/ 下：
%     *_load3-3_flat.mat  (3kg 居中, 平地)        —— 负载在体窗口
%     *_noload_flat.mat   (0kg 基线, 平地, --no_load) —— 全程(去前1s)
%   多 seed(1/2/3/42)按 variant 池化所有 per-env 值。
%
%   Figure 1: 俯仰/横滚 RMS 与速度跟踪误差，4 变体 × {0kg, 3kg} 分组条形(误差棒=池化 std)。
%   Figure 2: 负载退化量 (3kg − 0kg)，正值=加载后变差。
%
% per-env：每个环境一个 RMS 样本，跨 seed×env 池化，不先平均。

here = fileparts(mfilename('fullpath'));
ED = fullfile(here, '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
assert(exist(ED, 'dir') == 7, 'exported 不存在: %s', ED);
variants = {'Model-guided', 'Source-guided', 'Estimate-guided', 'RL-only'};
mets = {'pitch', 'roll', 'vxe'};
mlab = {'俯仰角 RMS [°]', '横滚角 RMS [°]', '速度跟踪误差 RMS [m/s]'};

% pool{vi, li}.(metric) = 向量（池化 per-env）；li: 1=0kg, 2=3kg
pool = cell(4, 2);
files = dir(fullfile(ED, 'play_data_*flat*.mat'));   % 同时匹配 *_load3-3_flat 与 *_flat_noload
for k = 1:numel(files)
    S = load(fullfile(files(k).folder, files(k).name));
    if ~isfield(S, 'meta'); continue; end
    fn = files(k).name;
    if contains(fn, 'noload'); li = 1; elseif contains(fn, 'load3-3'); li = 2; else; continue; end
    vi = find(strcmp(variants, variant_name(S.meta)), 1);
    if isempty(vi); continue; end
    dt = 0.02; if isfield(S, 'dt') && ~isempty(S.dt); dt = double(S.dt(1)); end
    ref = orient(S.payload_mass_ref_all);
    if li == 2
        msk = ref > 0.3;                  % 3kg: 负载在体
    else
        t = (0:size(ref, 1) - 1)' * dt; msk = repmat(t > 1.0, 1, size(ref, 2)); % 0kg: 全程去前1s
    end
    cmdx = orient(S.command_x_all); R2D = 180 / pi;
    vals.pitch = perenv_rms(orient(S.base_pitch_all), msk, dt) * R2D;
    vals.roll  = perenv_rms(orient(S.base_roll_all),  msk, dt) * R2D;
    vals.vxe   = perenv_rms(orient(S.base_lin_vel_x_all), msk, dt, cmdx);
    for mi = 1:numel(mets)
        if isempty(pool{vi, li}); pool{vi, li} = struct(); end
        f = mets{mi};
        if ~isfield(pool{vi, li}, f); pool{vi, li}.(f) = []; end
        pool{vi, li}.(f) = [pool{vi, li}.(f); vals.(f)(:)];
    end
end

getm = @(vi, li, f) localget(pool, vi, li, f);

% ---------- Figure 1: 0kg vs 3kg 分组条形 ----------
fig1 = figure('Color', 'w', 'Position', [60 60 1100 360], 'Name', 'Exp B 控制稳定性');
for mi = 1:numel(mets)
    subplot(1, 3, mi); hold on; grid on; box on;
    M = nan(4, 2); E = nan(4, 2);
    for vi = 1:4
        for li = 1:2
            v = getm(vi, li, mets{mi}); M(vi, li) = mean(v, 'omitnan'); E(vi, li) = std(v, 'omitnan');
        end
    end
    b = bar(M);
    ngroups = size(M, 1); nbars = size(M, 2); gw = min(0.8, nbars/(nbars+1.5));
    for li = 1:nbars
        x = (1:ngroups) - gw/2 + (2*li-1)*gw/(2*nbars);
        errorbar(x, M(:, li), E(:, li), 'k', 'linestyle', 'none', 'HandleVisibility', 'off');
    end
    set(gca, 'XTick', 1:4, 'XTickLabel', variants, 'XTickLabelRotation', 18);
    ylabel(mlab{mi}); legend({'0kg 基线', '3kg 负载'}, 'Location', 'northwest', 'FontSize', 8);
    title(mlab{mi});
end
sgtitle('实验B(平地)：3kg 居中负载下控制稳定性 vs 0kg 基线（多 seed 池化, 误差棒=std）', 'FontWeight', 'bold');
savefig_(fig1, fullfile(ED, 'expB_control_stability'));

% ---------- Figure 2: 负载退化量 (3kg − 0kg) ----------
fig2 = figure('Color', 'w', 'Position', [80 80 900 360], 'Name', 'Exp B 负载退化');
for mi = 1:numel(mets)
    subplot(1, 3, mi); hold on; grid on; box on;
    D = nan(4, 1);
    for vi = 1:4
        D(vi) = mean(getm(vi, 2, mets{mi}), 'omitnan') - mean(getm(vi, 1, mets{mi}), 'omitnan');
    end
    bar(D); yline(0, '-', 'Color', [.5 .5 .5]);
    set(gca, 'XTick', 1:4, 'XTickLabel', variants, 'XTickLabelRotation', 18);
    ylabel(['\Delta ' mlab{mi}]); title(['负载退化: ' mlab{mi}]);
end
sgtitle('实验B(平地)：负载引起的退化量 (3kg − 0kg)，越小越好', 'FontWeight', 'bold');
savefig_(fig2, fullfile(ED, 'expB_load_degradation'));

% ---------- 控制台 ----------
fprintf('\n==== 实验B 平地 控制稳定性（多 seed 池化 per-env, mean±std）====\n');
fprintf('%-16s | %-18s %-18s %-18s\n', '变体', 'pitch°(0/3kg)', 'roll°(0/3kg)', 'vx误差(0/3kg)');
for vi = 1:4
    s = sprintf('%-16s |', variants{vi});
    for mi = 1:numel(mets)
        v0 = getm(vi, 1, mets{mi}); v3 = getm(vi, 2, mets{mi});
        s = [s sprintf(' %.2f/%.2f      ', mean(v0, 'omitnan'), mean(v3, 'omitnan'))];
    end
    fprintf('%s\n', s);
end
end

% ================= 局部函数 =================
function v = localget(pool, vi, li, f)
v = [];
if ~isempty(pool{vi, li}) && isfield(pool{vi, li}, f); v = pool{vi, li}.(f); end
end

function A = orient(A)
A = double(A); if size(A, 1) < size(A, 2); A = A.'; end
end

function out = perenv_rms(A, segmask, dt, ref)
if nargin < 4; ref = []; end
ts = max(0, round(0.5 / dt)); N = size(A, 2); out = [];
for j = 1:N
    acc = [];
    for sg = seglist(segmask(:, j))'
        a = sg(1) + ts; b = sg(2);
        if a <= b
            x = A(a:b, j); if ~isempty(ref); x = x - ref(a:b, j); end
            acc = [acc; x]; %#ok<AGROW>
        end
    end
    if ~isempty(acc); out(end+1, 1) = sqrt(mean(acc.^2, 'omitnan')); end %#ok<AGROW>
end
end

function segs = seglist(mask)
mask = mask(:); d = diff([false; mask; false]);
segs = [find(d == 1), find(d == -1) - 1];   % [k x 2]
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

function savefig_(fig, base)
try
    exportgraphics(fig, [base '.pdf'], 'ContentType', 'vector');
    exportgraphics(fig, [base '.png'], 'Resolution', 150);
catch; saveas(fig, [base '.png']); end
fprintf('[expB] saved: %s.pdf/.png\n', base);
end
