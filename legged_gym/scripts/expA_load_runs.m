function [R, export_dir] = expA_load_runs(export_dir, settle_s)
% 实验A 共享数据加载：读取并分类 exported/ 下所有 play_data_*.mat。
%
% 输入：
%   export_dir  数据目录（默认 logs/wheelfoot_flat/WF_TRON1A/exported）
%   settle_s    每个开窗段开头剔除的"响应"时长（秒），默认 1.0。稳态估计 rl/qs
%               只统计每段去掉前 settle_s 之后的样本（时间序列 *_ts 不受影响）。
%
% 返回 R(i)：
%   .variant  'Model-guided' | 'Source-guided' | 'Estimate-guided' | 'RL-only'
%   .motion   'walk' | 'static'
%   .cond     'mass_sweep' | 'force_sweep' | 'mass_single' | 'force_single'
%   .ref      [1xN]  每个环境的参考真值（真实质量 或 力当量 kg）
%   .rl       [1xN]  每个环境 RL 编码器的稳态估计（去前 settle_s 的开窗均值）
%   .qs       [1xN]  每个环境 QS(Model-C) 的稳态估计
%   .t,.rl_ts,.qs_ts,.ref_ts  时间序列（跨环境平均；single 任务画时间图用，不剔除）
%   .file
%
% 分类完全依据 .mat 内的 meta。所有 per-env 量按环境分别保留，不跨环境平均。

if nargin < 1 || isempty(export_dir)
    here = fileparts(mfilename('fullpath'));
    export_dir = fullfile(here, '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
end
if nargin < 2 || isempty(settle_s); settle_s = 1.0; end
assert(exist(export_dir, 'dir') == 7, 'exported 目录不存在: %s', export_dir);
files = dir(fullfile(export_dir, 'play_data_*.mat'));
assert(~isempty(files), '在 %s 找不到 play_data_*.mat', export_dir);

R = struct([]);
for k = 1:numel(files)
    S = load(fullfile(files(k).folder, files(k).name));
    if ~isfield(S, 'meta'); continue; end
    rec = classify_and_extract(S, S.meta, settle_s);
    if isempty(rec); continue; end
    rec.file = files(k).name;
    if isempty(R); R = rec; else; R(end+1) = rec; end %#ok<AGROW>
end
fprintf('[expA] 载入 %d 个 run（来自 %s, 每段去前 %.1fs 响应）\n', numel(R), export_dir, settle_s);
end

% ------------------------------------------------------------------
function rec = classify_and_extract(S, meta, settle_s)
rec = [];
if nargin < 3 || isempty(settle_s); settle_s = 1.0; end
if ~isfield(S, 'encoder_mass_all') || ~isfield(S, 'payload_mass_all'); return; end
enc = double(S.encoder_mass_all); qs = double(S.payload_mass_all);  % [T,N]
if size(enc, 1) < size(enc, 2); enc = enc.'; end
if size(qs, 1)  < size(qs, 2);  qs  = qs.';  end
[T, N] = size(enc);
dt = 0.02; if isfield(S, 'dt') && ~isempty(S.dt); dt = double(S.dt(1)); end
t = (0:T-1)' * dt;

variant = variant_name(meta);
cmd_vx = getf(meta, 'cmd_vx', 0);
motion = 'static'; if abs(cmd_vx) > 1e-6; motion = 'walk'; end

start = getf(meta, 'load_start_time_s', 0.5);
dur   = getf(meta, 'load_duration_s_used', 6);
per   = getf(meta, 'load_interval_s_used', 10);

force_dir = 'none';
if getf(meta, 'ext_force_on', 0)
    if per > 0 && dur > 0
        onv = (t >= start) & (mod(t - start, per) < dur);
    else
        onv = (t >= start);
    end
    onmat = repmat(onv, 1, N);
    force_dir = lower(char(getf(meta, 'ext_force_dir', 'down')));
    if getf(meta, 'ext_force_sweep', 0)
        % 竖直 down 仍叫 force_sweep（兼容已有竖直实验）；水平等其它方向带后缀
        if strcmp(force_dir, 'down'); cond = 'force_sweep'; else; cond = ['force_sweep_' force_dir]; end
        ref = getf(meta, 'ext_force_equiv_kg_per_env', []); ref = double(ref(:)).';
        if numel(ref) ~= N; ref = nan(1, N); end
        ref_ts = double(onv) * median(ref(isfinite(ref)));
    else
        if strcmp(force_dir, 'down'); cond = 'force_single'; else; cond = ['force_single_' force_dir]; end
        fkg = getf(meta, 'ext_force_down_kg', 0);
        if fkg <= 0
            vec = getf(meta, 'ext_force_vec_N', [0 0 0]); fkg = norm(double(vec)) / 9.81;
        end
        ref = repmat(fkg, 1, N);
        ref_ts = double(onv) * fkg;
    end
else
    if ~isfield(S, 'payload_mass_ref_all'); return; end
    refall = double(S.payload_mass_ref_all);
    if size(refall, 1) < size(refall, 2); refall = refall.'; end
    onmat = refall > 0.3;
    lr = getf(meta, 'load_mass_range_used', [0 0]);
    if numel(lr) >= 2 && abs(lr(2) - lr(1)) > 1e-6; cond = 'mass_sweep'; else; cond = 'mass_single'; end
    ref = nan(1, N);
    for j = 1:N
        if any(onmat(:, j)); ref(j) = mean(refall(onmat(:, j), j), 'omitnan'); end
    end
    ref_ts = mean(refall, 2, 'omitnan');
end

% per-env 稳态估计：逐"开窗段"去掉开头 settle_s 的响应，再对剩余样本取均值
ts = max(0, round(settle_s / dt));
rl = nan(1, N); qse = nan(1, N);
for j = 1:N
    segs = seglist(onmat(:, j));   % [k x 2] 每个连续 on 段的 [start end]
    er = []; qr = [];
    for s = 1:size(segs, 1)
        a = segs(s, 1) + ts; b = segs(s, 2);
        if a <= b; er = [er; enc(a:b, j)]; qr = [qr; qs(a:b, j)]; end %#ok<AGROW>
    end
    if ~isempty(er); rl(j) = mean(er, 'omitnan'); qse(j) = mean(qr, 'omitnan'); end
end

rec = struct('variant', variant, 'motion', motion, 'cond', cond, 'force_dir', force_dir, ...
    'ref', ref, 'rl', rl, 'qs', qse, ...
    't', t, 'rl_ts', mean(enc, 2, 'omitnan'), 'qs_ts', mean(qs, 2, 'omitnan'), ...
    'ref_ts', ref_ts, 'file', '');
end

% ------------------------------------------------------------------
function vn = variant_name(meta)
% 变体判定与文件名 tag / Ch4 表4.1 一致：残差用 algorithm 的 use_load_residual_estimation
% （注意 env 的 use_residual_learning 不等价，Source-guided 那个为 1 会误导，勿用）。
qs = logical(getf(meta, 'use_qs_in_obs', true));
re = logical(getf(meta, 'use_load_residual_estimation', false));
tq = logical(getf(meta, 'use_torques_in_obs', true));
if qs && re
    vn = 'Model-guided';
elseif qs && ~re
    vn = 'Estimate-guided';
elseif ~qs && ~re && tq
    vn = 'Source-guided';
else
    vn = 'RL-only';
end
end

% ------------------------------------------------------------------
function segs = seglist(mask)
% 返回逻辑向量中每个连续 true 段的 [start end]（行索引），形状 [k x 2]。
mask = mask(:); d = diff([false; mask; false]);
starts = find(d == 1); ends = find(d == -1) - 1;
segs = [starts ends];
end

% ------------------------------------------------------------------
function v = getf(s, f, d)
if isstruct(s) && isfield(s, f) && ~isempty(s.(f)); v = s.(f); else; v = d; end
end
