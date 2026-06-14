function plot_training_curves(csvDir, outDir, smoothWindow, xMax)
%PLOT_TRAINING_CURVES Plot Ch4 2-4 kg training curves as mean +/- range.
%
% Usage:
%   plot_training_curves
%   plot_training_curves('logs/.../ch4_training_curves/ch4_narrow/csv')
%   plot_training_curves([], [], 1)       % raw curves, no moving average
%   plot_training_curves([], [], 101, 12000)

if nargin < 1 || isempty(csvDir)
    scriptDir = fileparts(mfilename('fullpath'));
    repoRoot = fullfile(scriptDir, '..', '..', '..');
    csvDir = fullfile(repoRoot, 'logs', 'wheelfoot_flat', 'WF_TRON1A', ...
        'exported_paper', 'ch4_training_curves', 'ch4_narrow', 'csv');
end
if nargin < 2 || isempty(outDir)
    outDir = fullfile(fileparts(csvDir), 'figures');
end
if nargin < 3 || isempty(smoothWindow)
    smoothWindow = 101;
end
if nargin < 4 || isempty(xMax)
    xMax = 12000;
end
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

csvPath = fullfile(csvDir, 'all_scalars_long.csv');
T = read_scalar_table(csvPath);
T.step = double(T.step);
T.value = double(T.value);

variants = {'Model-guided', 'Estimate-guided', 'Source-guided', 'RL-only'};
colors = containers.Map;
colors('Model-guided') = hex2rgb("E69F00");
colors('Estimate-guided') = hex2rgb("009E73");
colors('Source-guided') = hex2rgb("56B4E9");
colors('RL-only') = hex2rgb("D55E00");

metrics = {'mean_reward', 'mean_episode_length', 'payload_loss'};
ylabs = {'Mean reward', 'Mean episode length', 'Payload loss'};
titles = {'平均回报', '平均回合长度', 'Payload loss'};

fig = figure('Color', 'w', 'Position', [80 80 980 820], ...
    'Name', 'Ch4 常规负载训练曲线');
tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

for mi = 1:numel(metrics)
    nexttile; hold on; box off; grid on;
    metricRows = T(strcmp(T.metric, metrics{mi}), :);
    plotted = false;
    for vi = 1:numel(variants)
        variant = variants{vi};
        V = metricRows(strcmp(metricRows.variant, variant), :);
        if isempty(V)
            continue;
        end
        [x, yMean, yLo, yHi] = aggregate_runs(V, smoothWindow, xMax);
        if isempty(x)
            continue;
        end
        c = colors(variant);
        fill([x; flipud(x)], [yLo; flipud(yHi)], c, ...
            'FaceAlpha', 0.16, 'EdgeColor', 'none', ...
            'HandleVisibility', 'off');
        plot(x, yMean, 'LineWidth', 1.8, 'Color', c, ...
            'DisplayName', variant);
        plotted = true;
    end
    ylabel(ylabs{mi});
    title(titles{mi}, 'FontWeight', 'normal');
    xlim([0, xMax]);
    if mi == numel(metrics)
        xlabel('Training iteration');
    end
    if mi == 1 && plotted
        legend('Location', 'best', 'Box', 'off');
    end
    if ~plotted
        text(0.5, 0.5, sprintf('Missing TensorBoard tag for %s', metrics{mi}), ...
            'Units', 'normalized', 'HorizontalAlignment', 'center', ...
            'Color', [0.55 0.1 0.1]);
    end
end

sgtitle('2–4 kg 常规负载范围内各方案的训练曲线', 'FontWeight', 'bold');
save_outputs(fig, outDir, 'ch4_training_curves_mean_range');

end

function T = read_scalar_table(csvPath)
% Read robustly across MATLAB versions/import heuristics.
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
    if ~isempty(hit)
        names{hit} = expected{i};
    end
end
if width(T) == numel(expected) && ~all(ismember(expected, names))
    names = expected;
end
T.Properties.VariableNames = names;

missing = expected(~ismember(expected, T.Properties.VariableNames));
if ~isempty(missing)
    fprintf('CSV path: %s\n', csvPath);
    fprintf('Detected variables:\n');
    disp(T.Properties.VariableNames);
    error('Missing expected CSV columns: %s', strjoin(missing, ', '));
end
end

function [x, yMean, yLo, yHi] = aggregate_runs(V, smoothWindow, xMax)
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
yMean = mean(Y, 2, 'omitnan');
yLo = min(Y, [], 2, 'omitnan');
yHi = max(Y, [], 2, 'omitnan');
valid = ~isnan(yMean);
x = x(valid);
yMean = yMean(valid);
yLo = yLo(valid);
yHi = yHi(valid);
end

function save_outputs(fig, outDir, stem)
pngPath = fullfile(outDir, [stem '.png']);
pdfPath = fullfile(outDir, [stem '.pdf']);
figPath = fullfile(outDir, [stem '.fig']);
savefig(fig, figPath);
exportgraphics(fig, pngPath, 'Resolution', 300);
exportgraphics(fig, pdfPath, 'ContentType', 'vector');
fprintf('[plot_training_curves] saved:\n  %s\n  %s\n  %s\n', pngPath, pdfPath, figPath);
end

function rgb = hex2rgb(hex)
hex = char(erase(string(hex), '#'));
rgb = [hex2dec(hex(1:2)), hex2dec(hex(3:4)), hex2dec(hex(5:6))] ./ 255;
end
