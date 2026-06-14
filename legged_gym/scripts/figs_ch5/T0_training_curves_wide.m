function T0_training_curves_wide(smoothWindow, xMax)
% T0 宽负载(2-30kg)训练曲线:3 子图 reward / episode length / payload loss。
% 每方法两条 seed 先平滑,再画均值曲线;阴影=两个 seed 的 min-max 范围。
% 数据来自 TensorBoard 导出的 ch5_wide CSV。用法:T0_training_curves_wide(101, 11000)。不自动保存。
if nargin<1 || isempty(smoothWindow); smoothWindow = 101; end
if nargin<2 || isempty(xMax); xMax = 11000; end

here = fileparts(mfilename('fullpath'));
csvDir = fullfile(here, '..', '..', '..', 'logs', 'wheelfoot_flat', 'WF_TRON1A', ...
    'exported_paper', 'ch5_training_curves', 'ch5_wide', 'csv');
csvPath = fullfile(csvDir, 'all_scalars_long.csv');

T = read_scalar_table(csvPath);
T.step = double(T.step);
T.value = double(T.value);

variants = {'Model-guided', 'Estimate-guided', 'Source-guided', 'RL-only'};
col = ch5_colors();
metrics = {'mean_reward', 'mean_episode_length', 'payload_loss'};
ylabs = {'Mean reward', 'Mean episode length', 'Payload loss'};
titles = {'平均回报', '平均回合长度', 'Payload loss'};

figure('Color','w','Position',[60 50 760 760],'Name','T0 宽负载训练曲线');
for mi = 1:numel(metrics)
    ax = subplot(3,1,mi); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    metricRows = T(strcmp(T.metric, metrics{mi}), :);
    h = gobjects(0); lab = {};
    for vi = 1:numel(variants)
        V = metricRows(strcmp(metricRows.variant, variants{vi}), :);
        if isempty(V); continue; end
        [x, yMean, yLo, yHi, nRuns] = aggregate_runs(V, smoothWindow, xMax, 2);
        if isempty(x); continue; end
        band_fill(ax, x, yLo, yHi, col{vi});
        h(end+1) = plot(ax, x, yMean, '-', 'Color', col{vi}, 'LineWidth', 2.2); %#ok<AGROW>
        lab{end+1} = variants{vi}; %#ok<AGROW>
    end
    ylabel(ax, ylabs{mi});
    title(ax, titles{mi}, 'FontWeight','bold');
    xlim(ax, [0 xMax]);
    if mi==3; xlabel(ax, 'Training iteration'); end
    if mi==1 && ~isempty(h); legend(ax, h, lab, 'Location','best'); end
end
sgtitle('2–30 kg 负载范围内各方案的训练曲线', 'FontWeight','bold');
end

function T = read_scalar_table(csvPath)
try
    opts = detectImportOptions(csvPath, 'FileType', 'text', 'Delimiter', ',');
    opts.VariableNamingRule = 'preserve';
    T = readtable(csvPath, opts, 'TextType', 'string');
catch
    T = readtable(csvPath, 'FileType', 'text', 'Delimiter', ',', ...
        'ReadVariableNames', true, 'TextType', 'string');
end

expected = {'variant', 'run_name', 'metric', 'tag', 'wall_time', 'step', 'value'};
names = T.Properties.VariableNames;
clean = lower(regexprep(names, '[^a-zA-Z0-9]', ''));
for i = 1:numel(expected)
    key = lower(regexprep(expected{i}, '[^a-zA-Z0-9]', ''));
    hit = find(strcmp(clean, key), 1);
    if ~isempty(hit); names{hit} = expected{i}; end
end
if width(T) == numel(expected) && ~all(ismember(expected, names))
    names = expected;
end
T.Properties.VariableNames = names;

missing = expected(~ismember(expected, T.Properties.VariableNames));
if ~isempty(missing)
    fprintf('CSV path: %s\n', csvPath);
    fprintf('Detected variables:\n'); disp(T.Properties.VariableNames);
    error('Missing expected CSV columns: %s', strjoin(missing, ', '));
end
end

function [x, yMean, yLo, yHi, nRuns] = aggregate_runs(V, smoothWindow, xMax, minRuns)
runs = unique(V.run_name, 'stable');
x = unique(V.step);
x = x(x >= 0 & x <= xMax);
Y = nan(numel(x), numel(runs));
for ri = 1:numel(runs)
    R = sortrows(V(strcmp(V.run_name, runs(ri)), :), 'step');
    [rs, keep] = unique(R.step, 'stable');
    rv = R.value(keep);
    Y(:, ri) = interp1(rs, rv, x, 'linear', nan);
end
if smoothWindow > 1
    Y = movmean(Y, smoothWindow, 1, 'omitnan');
end
nValid = sum(~isnan(Y), 2);
yMean = mean(Y, 2, 'omitnan');
yLo = min(Y, [], 2, 'omitnan');
yHi = max(Y, [], 2, 'omitnan');
valid = ~isnan(yMean) & nValid >= minRuns;
x = x(valid); yMean = yMean(valid); yLo = yLo(valid); yHi = yHi(valid);
nRuns = numel(runs);
end

function band_fill(ax, x, yLo, yHi, c)
m = isfinite(x) & isfinite(yLo) & isfinite(yHi);
if ~any(m); return; end
xx = x(m); a = yLo(m); b = yHi(m);
fill(ax, [xx; flipud(xx)], [a; flipud(b)], c, ...
    'FaceAlpha', 0.12, 'EdgeColor','none', 'HandleVisibility','off');
end
