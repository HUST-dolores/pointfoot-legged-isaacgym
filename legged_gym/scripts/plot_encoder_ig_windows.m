function plot_encoder_ig_windows(inputPath, targetName, topN, outputImagePath)
%PLOT_ENCODER_IG_WINDOWS Plot rollout-time Integrated Gradients heatmap.
%
% Usage:
%   plot_encoder_ig_windows("encoder_ig_tables_20260521_134921.mat")
%   plot_encoder_ig_windows("encoder_ig_tables_20260521_134921.mat", "com")
%   plot_encoder_ig_windows("encoder_ig_windows_20260521_134921.csv", "all", 12)
%   plot_encoder_ig_windows(path, "mass", 16, "mass_heatmap.png")
%
% The heatmap x-axis is rollout time, y-axis is observation group, and color is
% the within-window attribution percentage.

if nargin < 1 || strlength(string(inputPath)) == 0
    [fileName, folderName] = uigetfile({'*.mat;*.csv', 'IG tables (*.mat, *.csv)'});
    if isequal(fileName, 0)
        return;
    end
    inputPath = fullfile(folderName, fileName);
end

if nargin < 2 || strlength(string(targetName)) == 0
    targetName = "all";
end

if nargin < 3 || isempty(topN)
    topN = inf;
end

if nargin < 4
    outputImagePath = "";
end

T = load_window_table(inputPath);
targetName = string(targetName);
T = T(string(T.target) == targetName, :);

if isempty(T)
    error("No rows found for target '%s'.", targetName);
end

windowIds = unique(double(T.window_index), "stable");
labels = unique(string(T.label), "stable");

Z = nan(numel(labels), numel(windowIds));
timeStart = nan(1, numel(windowIds));
timeEnd = nan(1, numel(windowIds));

for c = 1:numel(windowIds)
    win = windowIds(c);
    winMask = double(T.window_index) == win;
    timeStart(c) = first_numeric(T.time_start_s(winMask));
    timeEnd(c) = first_numeric(T.time_end_s(winMask));

    for r = 1:numel(labels)
        rowMask = winMask & (string(T.label) == labels(r));
        if any(rowMask)
            Z(r, c) = first_numeric(T.percent(rowMask));
        end
    end
end

rowMean = mean(Z, 2, "omitnan");
[~, order] = sort(rowMean, "descend");
if isfinite(topN)
    order = order(1:min(numel(order), topN));
end
Z = Z(order, :);
labels = labels(order);

timeLabels = strings(1, numel(windowIds));
for c = 1:numel(windowIds)
    timeLabels(c) = sprintf("%.2f-%.2fs", timeStart(c), timeEnd(c));
end

figure("Color", "w", "Name", "Encoder IG window heatmap");
imagesc(Z);
axis tight;
colormap(parula);
cb = colorbar;
cb.Label.String = "Attribution (%)";

ax = gca;
ax.XTick = 1:numel(windowIds);
ax.XTickLabel = timeLabels;
ax.XTickLabelRotation = 45;
ax.YTick = 1:numel(labels);
ax.YTickLabel = labels;
ax.TickLabelInterpreter = "none";

xlabel("Rollout time window");
ylabel("Observation group");
title(sprintf("Encoder IG importance over time, target = %s", targetName), ...
    "Interpreter", "none");

set(gca, "FontName", "Arial", "FontSize", 10);

if strlength(string(outputImagePath)) > 0
    exportgraphics(gcf, outputImagePath, "Resolution", 200);
end
end

function T = load_window_table(inputPath)
[~, ~, ext] = fileparts(inputPath);
ext = lower(string(ext));

if ext == ".csv"
    T = readtable(inputPath);
    return;
end

if ext ~= ".mat"
    error("Expected a .mat or .csv file, got: %s", inputPath);
end

S = load(inputPath);
if ~isfield(S, "window_cell")
    error("MAT file does not contain window_cell.");
end

C = S.window_cell;
if isempty(C) || size(C, 1) < 2
    error("window_cell is empty. Re-run analysis with --window_steps > 0.");
end

header = matlab.lang.makeValidName(string(C(1, :)));
T = cell2table(C(2:end, :), "VariableNames", cellstr(header));
T = convert_numeric_columns(T);
end

function T = convert_numeric_columns(T)
numericColumns = [
    "window_index"
    "analysis_step_start"
    "analysis_step_end"
    "rollout_step_start"
    "rollout_step_end"
    "time_start_s"
    "time_end_s"
    "num_candidate_samples"
    "num_ig_samples"
    "target_total_attribution"
    "feature_dim"
    "attribution"
    "percent"
];

for i = 1:numel(numericColumns)
    name = numericColumns(i);
    if ismember(name, string(T.Properties.VariableNames))
        if iscell(T.(name))
            T.(name) = cellfun(@double, T.(name));
        else
            T.(name) = double(T.(name));
        end
    end
end
end

function value = first_numeric(values)
if istable(values)
    values = values{:, 1};
end
if iscell(values)
    value = double(values{1});
else
    value = double(values(1));
end
end
