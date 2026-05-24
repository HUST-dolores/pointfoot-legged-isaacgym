function T = plot_payload_experiments(dataRoot, outDir, saveOutputs, plotMode)
%PLOT_PAYLOAD_EXPERIMENTS Summarize and plot payload-estimation play .mat files.
%
% Usage from MATLAB, at the repository root:
%   addpath('legged_gym/scripts')
%   T = plot_payload_experiments
%
% Or specify paths explicitly:
%   T = plot_payload_experiments('logs/wheelfoot_flat/WF_TRON1A/exported')
%
% By default this only builds the summary table and does not plot.  Pass a
% plotMode when you want a coarse overview:
%   T = plot_payload_experiments([], [], false, 'estimation')
%   T = plot_payload_experiments([], [], false, 'control')
%   T = plot_payload_experiments([], [], false, 'timeseries')
%   T = plot_payload_experiments([], [], false, 'scatter')
%   T = plot_payload_experiments([], [], false, 'all')
%
% Figures stay open unless saveOutputs=true.
%
% The script is intentionally self-contained. It scans play_data_*.mat files,
% recomputes metrics from *_all arrays when available, writes a summary table,
% and saves publication-oriented PNG/PDF figures.

if nargin < 1 || isempty(dataRoot)
    dataRoot = fullfile('logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported');
end
if nargin < 2 || isempty(outDir)
    outDir = fullfile(dataRoot, 'paper_figures_payload');
end
if nargin < 3 || isempty(saveOutputs)
    saveOutputs = false;
end
if nargin < 4 || isempty(plotMode)
    plotMode = 'none';
end
if saveOutputs && exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

files = dir(fullfile(dataRoot, 'play_data_*.mat'));
files = files(~strcmp({files.name}, 'play_data.mat'));
if isempty(files)
    error('No play_data_*.mat files found under %s', dataRoot);
end

records = struct([]);
for i = 1:numel(files)
    path = fullfile(files(i).folder, files(i).name);
    S = load(path);
    rec = summarize_one_mat(S, path, files(i).name);
    records = [records; rec]; %#ok<AGROW>
end

T = struct2table(records);
T = sortrows(T, {'condition', 'load_range', 'method', 'seed', 'timestamp'});

summaryCsv = fullfile(outDir, 'payload_summary_table.csv');
summaryMat = fullfile(outDir, 'payload_summary_table.mat');
if saveOutputs
    writetable(T, summaryCsv);
    save(summaryMat, 'T');
end

switch lower(plotMode)
    case 'none'
        % Summary table only.
    case 'estimation'
        plot_estimation_bars(T, outDir, saveOutputs);
    case 'control'
        plot_control_bars(T, outDir, saveOutputs);
    case 'condition_bars'
        plot_method_condition_bars(T, outDir, saveOutputs);
    case 'timeseries'
        maxExamples = min(4, height(T));
        for i = 1:maxExamples
            plot_one_timeseries(T.mat_path{i}, outDir, saveOutputs);
        end
    case 'scatter'
        maxExamples = min(4, height(T));
        for i = 1:maxExamples
            plot_one_scatter(T.mat_path{i}, outDir, saveOutputs);
        end
    case 'all'
        plot_estimation_bars(T, outDir, saveOutputs);
        plot_control_bars(T, outDir, saveOutputs);
        plot_method_condition_bars(T, outDir, saveOutputs);
        maxExamples = min(4, height(T));
        for i = 1:maxExamples
            plot_one_timeseries(T.mat_path{i}, outDir, saveOutputs);
            plot_one_scatter(T.mat_path{i}, outDir, saveOutputs);
        end
    otherwise
        error('Unknown plotMode: %s', plotMode);
end

if saveOutputs
    fprintf('Saved summary: %s\n', summaryCsv);
    fprintf('Saved figures: %s\n', outDir);
else
    if strcmpi(plotMode, 'none')
        fprintf('Summary-only mode: no figures were drawn. T contains the summary table.\n');
    else
        fprintf('Preview mode: figures are open and were not saved. T contains the summary table.\n');
    end
end
end


function rec = summarize_one_mat(S, path, fileName)
meta = parse_filename(fileName);
dt = get_scalar(S, 'dt', 0.02);

loadMask = get_mat(S, 'load_on_body_all');
if isempty(loadMask)
    refMass = get_mat(S, 'payload_mass_ref_all');
    loadMask = abs(refMass) > 1.0e-6;
else
    loadMask = loadMask > 0.5;
end

motionMask = true(size(loadMask));
baseVx = get_mat(S, 'base_lin_vel_x_all');
if ~isempty(baseVx)
    motionMask = abs(baseVx) > 0.15;
end

rec = struct();
rec.file = char(fileName);
rec.mat_path = char(path);
rec.timestamp = char(meta.timestamp);
rec.method = char(meta.method);
rec.condition = char(meta.condition);
rec.seed = meta.seed;
rec.checkpoint = meta.checkpoint;
rec.load_min = meta.load_min;
rec.load_max = meta.load_max;
rec.load_mid = meta.load_mid;
rec.load_range = char(meta.load_range);
rec.label = char(meta.label);
rec.dt = dt;
rec.num_samples_loaded = nnz(loadMask);
rec.active_frac = mean(loadMask(:), 'omitnan');

% Estimation metrics. QS means Model C analytical estimate; ENC means the
% encoder output already converted to physical units by play.py.
rec.qs_mass_rmse_kg = rmse_field(S, 'payload_mass_all', 'payload_mass_ref_all', loadMask);
rec.enc_mass_rmse_kg = rmse_field(S, 'encoder_mass_all', 'payload_mass_ref_all', loadMask);
rec.qs_mass_bias_kg = bias_field(S, 'payload_mass_all', 'payload_mass_ref_all', loadMask);
rec.enc_mass_bias_kg = bias_field(S, 'encoder_mass_all', 'payload_mass_ref_all', loadMask);

rec.qs_load_x_rmse_m = rmse_field(S, 'load_x_all', 'load_x_ref_all', loadMask);
rec.qs_load_y_rmse_m = rmse_field(S, 'load_y_all', 'load_y_ref_all', loadMask);

rec.qs_dcom_x_rmse_m = rmse_field(S, 'modelC_com_delta_x_all', 'true_com_delta_x_all', loadMask);
rec.qs_dcom_y_rmse_m = rmse_field(S, 'modelC_com_delta_y_all', 'true_com_delta_y_all', loadMask);
rec.enc_dcom_x_rmse_m = rmse_field(S, 'encoder_com_delta_x_all', 'true_com_delta_x_all', loadMask);
rec.enc_dcom_y_rmse_m = rmse_field(S, 'encoder_com_delta_y_all', 'true_com_delta_y_all', loadMask);
rec.qs_dcom_x_bias_m = bias_field(S, 'modelC_com_delta_x_all', 'true_com_delta_x_all', loadMask);
rec.qs_dcom_y_bias_m = bias_field(S, 'modelC_com_delta_y_all', 'true_com_delta_y_all', loadMask);
rec.enc_dcom_x_bias_m = bias_field(S, 'encoder_com_delta_x_all', 'true_com_delta_x_all', loadMask);
rec.enc_dcom_y_bias_m = bias_field(S, 'encoder_com_delta_y_all', 'true_com_delta_y_all', loadMask);

rec.qs_mass_dyn_rmse_kg = rmse_field(S, 'payload_mass_all', 'payload_mass_ref_all', loadMask & motionMask);
rec.enc_mass_dyn_rmse_kg = rmse_field(S, 'encoder_mass_all', 'payload_mass_ref_all', loadMask & motionMask);

% Control metrics. Some fields are not logged yet in older .mat files; these
% become NaN and will appear as gaps in the plots.
rec.lin_vx_tracking_rmse = rmse_field(S, 'base_lin_vel_x_all', 'command_x_all', true(size(loadMask)));
rec.yaw_tracking_rmse = rmse_field(S, 'base_ang_vel_z_all', 'command_yaw_all', true(size(loadMask)));
rec.power_mean = mean_field(S, 'power', []);
rec.torque_rms = rms_concat_fields(S, { ...
    'torque_abad_L', 'torque_hip_L', 'torque_knee_L', 'torque_wheel_L', ...
    'torque_abad_R', 'torque_hip_R', 'torque_knee_R', 'torque_wheel_R'});

% Event-style metrics for future perturbation experiments. These detect jumps
% in the mean true load mass/position and align errors after the first jump.
event = event_metrics(S, dt);
rec.event_time_s = event.time_s;
rec.event_qs_mass_peak_kg = event.qs_mass_peak_kg;
rec.event_enc_mass_peak_kg = event.enc_mass_peak_kg;
rec.event_qs_mass_post_rmse_kg = event.qs_mass_post_rmse_kg;
rec.event_enc_mass_post_rmse_kg = event.enc_mass_post_rmse_kg;
end


function meta = parse_filename(fileName)
name = regexprep(char(fileName), '\.mat$', '');
meta.timestamp = '';
tok = regexp(name, 'play_data_(\d{8}-\d{6})', 'tokens', 'once');
if ~isempty(tok), meta.timestamp = tok{1}; end

if has_text(name, 'qs1_resid1')
    meta.method = 'qs_residual';
elseif has_text(name, 'qs0_resid0')
    meta.method = 'history_only';
elseif has_text(name, 'qs1_resid0')
    meta.method = 'qs_direct';
elseif has_text(lower(name), 'oracle')
    meta.method = 'oracle';
elseif has_text(lower(name), 'no_load')
    meta.method = 'no_load_info';
else
    meta.method = 'unknown';
end

lname = lower(name);
if has_text(lname, 'static')
    meta.condition = 'static';
elseif has_text(lname, 'walk')
    meta.condition = 'walk';
elseif has_text(lname, 'push')
    meta.condition = 'push';
elseif has_text(lname, 'accel')
    meta.condition = 'accel';
elseif has_text(lname, 'grid')
    meta.condition = 'grid';
elseif has_text(lname, 'terrain')
    meta.condition = 'terrain';
else
    meta.condition = 'unknown';
end

meta.seed = parse_number(name, 'seed(\d+)', NaN);
meta.checkpoint = parse_number(name, 'ckpt(\d+)', NaN);
loadTok = regexp(name, 'load([0-9.]+)-([0-9.]+)', 'tokens', 'once');
if isempty(loadTok)
    meta.load_min = NaN;
    meta.load_max = NaN;
    meta.load_mid = NaN;
    meta.load_range = 'unknown';
else
    meta.load_min = str2double(loadTok{1});
    meta.load_max = str2double(loadTok{2});
    meta.load_mid = 0.5 * (meta.load_min + meta.load_max);
    meta.load_range = sprintf('%g-%g', meta.load_min, meta.load_max);
end
meta.label = sprintf('%s | %s | load %s | seed %g', ...
    meta.method, meta.condition, meta.load_range, meta.seed);
end


function tf = has_text(str, pat)
tf = ~isempty(strfind(char(str), char(pat))); %#ok<STREMP>
end


function x = parse_number(str, pattern, defaultValue)
tok = regexp(str, pattern, 'tokens', 'once');
if isempty(tok)
    x = defaultValue;
else
    x = str2double(tok{1});
end
end


function A = get_mat(S, field)
if isfield(S, field)
    A = double(S.(field));
    A = squeeze(A);
    if isvector(A)
        A = A(:);
    end
else
    A = [];
end
end


function x = get_scalar(S, field, defaultValue)
if isfield(S, field)
    v = double(S.(field));
    x = v(1);
else
    x = defaultValue;
end
end


function v = rmse_field(S, estField, refField, mask)
est = get_mat(S, estField);
ref = get_mat(S, refField);
if isempty(est) || isempty(ref) || ~isequal(size(est), size(ref))
    v = NaN;
    return;
end
if isscalar(mask)
    mask = true(size(est));
end
if ~isequal(size(mask), size(est))
    mask = true(size(est));
end
valid = mask & isfinite(est) & isfinite(ref);
if ~any(valid(:))
    v = NaN;
else
    err = est(valid) - ref(valid);
    v = sqrt(mean(err.^2, 'omitnan'));
end
end


function v = bias_field(S, estField, refField, mask)
est = get_mat(S, estField);
ref = get_mat(S, refField);
if isempty(est) || isempty(ref) || ~isequal(size(est), size(ref))
    v = NaN;
    return;
end
if isscalar(mask)
    mask = true(size(est));
end
if ~isequal(size(mask), size(est))
    mask = true(size(est));
end
valid = mask & isfinite(est) & isfinite(ref);
if ~any(valid(:))
    v = NaN;
else
    err = est(valid) - ref(valid);
    v = mean(err, 'omitnan');
end
end


function v = mean_field(S, field, mask)
A = get_mat(S, field);
if isempty(A)
    v = NaN;
    return;
end
if isempty(mask)
    valid = isfinite(A);
elseif isequal(size(mask), size(A))
    valid = mask & isfinite(A);
else
    valid = isfinite(A);
end
if ~any(valid(:))
    v = NaN;
else
    v = mean(A(valid), 'omitnan');
end
end


function v = rms_concat_fields(S, fields)
allVals = [];
for i = 1:numel(fields)
    A = get_mat(S, fields{i});
    if ~isempty(A)
        allVals = [allVals; A(:)]; %#ok<AGROW>
    end
end
allVals = allVals(isfinite(allVals));
if isempty(allVals)
    v = NaN;
else
    v = sqrt(mean(allVals.^2));
end
end


function event = event_metrics(S, dt)
event = struct( ...
    'time_s', NaN, ...
    'qs_mass_peak_kg', NaN, ...
    'enc_mass_peak_kg', NaN, ...
    'qs_mass_post_rmse_kg', NaN, ...
    'enc_mass_post_rmse_kg', NaN);

ref = get_mat(S, 'payload_mass_ref_all');
if isempty(ref) || size(ref, 1) < 5
    return;
end
meanRef = mean(ref, 2, 'omitnan');
d = abs(diff(meanRef));
thr = max(0.5, 5 * median(d, 'omitnan'));
[peakJump, idx] = max(d);
if isempty(idx) || peakJump < thr
    return;
end
event.time_s = idx * dt;
win = idx:min(size(ref, 1), idx + max(1, round(2.0 / dt)));
post = min(size(ref, 1), idx + max(1, round(5.0 / dt))): ...
       min(size(ref, 1), idx + max(1, round(10.0 / dt)));
mask = true(size(ref));

event.qs_mass_peak_kg = peak_abs_error(S, 'payload_mass_all', 'payload_mass_ref_all', mask, win);
event.enc_mass_peak_kg = peak_abs_error(S, 'encoder_mass_all', 'payload_mass_ref_all', mask, win);
event.qs_mass_post_rmse_kg = rmse_over_rows(S, 'payload_mass_all', 'payload_mass_ref_all', post);
event.enc_mass_post_rmse_kg = rmse_over_rows(S, 'encoder_mass_all', 'payload_mass_ref_all', post);
end


function v = peak_abs_error(S, estField, refField, mask, rows)
est = get_mat(S, estField);
ref = get_mat(S, refField);
if isempty(est) || isempty(ref) || ~isequal(size(est), size(ref))
    v = NaN;
    return;
end
rows = rows(rows >= 1 & rows <= size(est, 1));
if isempty(rows)
    v = NaN;
    return;
end
valid = mask(rows, :) & isfinite(est(rows, :)) & isfinite(ref(rows, :));
err = abs(est(rows, :) - ref(rows, :));
if ~any(valid(:))
    v = NaN;
else
    v = max(err(valid));
end
end


function v = rmse_over_rows(S, estField, refField, rows)
est = get_mat(S, estField);
ref = get_mat(S, refField);
if isempty(est) || isempty(ref) || ~isequal(size(est), size(ref))
    v = NaN;
    return;
end
rows = rows(rows >= 1 & rows <= size(est, 1));
if isempty(rows)
    v = NaN;
    return;
end
err = est(rows, :) - ref(rows, :);
v = sqrt(mean(err(:).^2, 'omitnan'));
end


function plot_estimation_bars(T, outDir, saveOutputs)
print_data_sources(T, 'estimation_rmse_bars');
plotLabels = matlab.lang.makeUniqueStrings(table_text_column(T, 'label'));
labels = categorical(plotLabels);
labels = reordercats(labels, plotLabels);
methodCol = table_text_column(T, 'method');
encoderColors = payload_method_color(methodCol, 'encoder');
qsColors = payload_method_color(methodCol, 'qs');

fig = figure('Color', 'w', 'Position', [80, 80, 1300, 760]);

subplot(3, 1, 1);
bar_qs_encoder(labels, [T.qs_mass_rmse_kg, T.enc_mass_rmse_kg], qsColors, encoderColors);
ylabel('Mass RMSE [kg]');
legend({'QS', 'Encoder'}, 'Location', 'northoutside', 'Orientation', 'horizontal');
title('Payload Mass Estimation');
grid on;

subplot(3, 1, 2);
bar_qs_encoder(labels, 100 * [T.qs_dcom_x_rmse_m, T.enc_dcom_x_rmse_m], qsColors, encoderColors);
ylabel('CoM-x RMSE [cm]');
legend({'QS', 'Encoder'}, 'Location', 'northoutside', 'Orientation', 'horizontal');
title('Total CoM Delta X');
grid on;

subplot(3, 1, 3);
bar_qs_encoder(labels, 100 * [T.qs_dcom_y_rmse_m, T.enc_dcom_y_rmse_m], qsColors, encoderColors);
ylabel('CoM-y RMSE [cm]');
legend({'QS', 'Encoder'}, 'Location', 'northoutside', 'Orientation', 'horizontal');
title('Total CoM Delta Y');
grid on;
rotate_xticks(fig, 35);
finish_figure(fig, outDir, 'estimation_rmse_bars', saveOutputs);
end


function bar_qs_encoder(x, y, qsColors, encoderColors)
h = bar(x, y);
try
    h(1).FaceColor = 'flat';
    h(1).CData = qsColors;
    h(2).FaceColor = 'flat';
    h(2).CData = encoderColors;
catch
end
end


function plot_control_bars(T, outDir, saveOutputs)
print_data_sources(T, 'control_metrics_bars');
plotLabels = matlab.lang.makeUniqueStrings(table_text_column(T, 'label'));
labels = categorical(plotLabels);
labels = reordercats(labels, plotLabels);
methodColors = payload_method_color(table_text_column(T, 'method'));
fig = figure('Color', 'w', 'Position', [120, 120, 1300, 620]);

subplot(3, 1, 1);
bar_with_colors(labels, T.lin_vx_tracking_rmse, methodColors);
ylabel('v_x RMSE [m/s]');
title('Linear Velocity Tracking');
grid on;

subplot(3, 1, 2);
bar_with_colors(labels, T.yaw_tracking_rmse, methodColors);
ylabel('yaw RMSE [rad/s]');
title('Yaw Rate Tracking');
grid on;

subplot(3, 1, 3);
bar(labels, [T.power_mean, T.torque_rms]);
ylabel('cost');
legend({'mean power', 'joint torque RMS'}, 'Location', 'northoutside', 'Orientation', 'horizontal');
title('Control Cost Proxies');
grid on;
rotate_xticks(fig, 35);
finish_figure(fig, outDir, 'control_metrics_bars', saveOutputs);
end


function plot_method_condition_bars(T, outDir, saveOutputs)
% Compact replacements for the old heatmaps.  They average repeated rows
% within each method/condition or method/load group and show them as bars.
plot_grouped_metric_bars(T, 'method', 'condition', 'enc_mass_rmse_kg', ...
    'Encoder mass RMSE by condition', 'mass RMSE [kg]', outDir, ...
    'bar_encoder_mass_by_condition', saveOutputs);
plot_grouped_metric_bars(T, 'method', 'condition', 'qs_mass_rmse_kg', ...
    'QS mass RMSE by condition', 'mass RMSE [kg]', outDir, ...
    'bar_qs_mass_by_condition', saveOutputs);
plot_grouped_metric_bars(T, 'method', 'load_range', 'enc_mass_rmse_kg', ...
    'Encoder mass RMSE by load range', 'mass RMSE [kg]', outDir, ...
    'bar_encoder_mass_by_load', saveOutputs);
plot_qs_vs_encoder_mass_bars(T, outDir, saveOutputs);
end


function bar_with_colors(x, y, colors)
b = bar(x, y);
try
    b.FaceColor = 'flat';
    b.CData = colors;
catch
end
end


function plot_grouped_metric_bars(T, rowVar, colVar, valueVar, figTitle, yLabelText, outDir, figName, saveOutputs)
print_data_sources(T, figName);
rowVals = table_text_column(T, rowVar);
colVals = table_text_column(T, colVar);
rows = unique(rowVals, 'stable');
cols = unique(colVals, 'stable');
M = NaN(numel(rows), numel(cols));
for i = 1:numel(rows)
    for j = 1:numel(cols)
        mask = strcmp(rowVals, rows{i}) & strcmp(colVals, cols{j});
        vals = T.(valueVar)(mask);
        if ~isempty(vals)
            M(i, j) = mean(vals, 'omitnan');
        end
    end
end
print_missing_groups(rows, cols, M, figName);
fig = figure('Color', 'w', 'Position', [180, 180, 980, 520]);
if strcmp(rowVar, 'method')
    xLabels = categorical(cols);
    xLabels = reordercats(xLabels, cols);
    h = bar(xLabels, M');
    for k = 1:numel(h)
        try
            h(k).FaceColor = payload_method_color(rows{k});
        catch
        end
    end
    legend(rows, 'Location', 'northoutside', 'Orientation', 'horizontal', 'Interpreter', 'none');
else
    bar(categorical(rows), M);
    legend(cols, 'Location', 'northoutside', 'Orientation', 'horizontal', 'Interpreter', 'none');
end
ylabel(yLabelText);
title(figTitle);
grid on;
rotate_xticks(fig, 25);
finish_figure(fig, outDir, figName, saveOutputs);
end


function print_missing_groups(rows, cols, M, figName)
missing = isnan(M);
if ~any(missing(:))
    return;
end
fprintf('[missing groups] %s\n', figName);
for i = 1:numel(rows)
    for j = 1:numel(cols)
        if missing(i, j)
            fprintf('  no data: %s / %s\n', rows{i}, cols{j});
        end
    end
end
end


function plot_qs_vs_encoder_mass_bars(T, outDir, saveOutputs)
% Fair compact comparison: use the common dataset available for all current
% methods (load 2-4 kg, seed 42) and aggregate repeated plays.
methodCol = table_text_column(T, 'method');
condCol = table_text_column(T, 'condition');
loadCol = table_text_column(T, 'load_range');
mask = strcmp(loadCol, '2-4') & (T.seed == 42);
Tfair = T(mask, :);
if isempty(Tfair)
    return;
end
print_data_sources(Tfair, 'bar_qs_vs_encoder_mass_common_load2_4_seed42');

methodCol = table_text_column(Tfair, 'method');
condCol = table_text_column(Tfair, 'condition');
pairLabels = cell(height(Tfair), 1);
for i = 1:height(Tfair)
    pairLabels{i} = sprintf('%s | %s', condCol{i}, methodCol{i});
end
pairs = unique(pairLabels, 'stable');
M = NaN(numel(pairs), 2);
pairMethods = cell(numel(pairs), 1);
for i = 1:numel(pairs)
    m = strcmp(pairLabels, pairs{i});
    M(i, 1) = mean(Tfair.qs_mass_rmse_kg(m), 'omitnan');
    M(i, 2) = mean(Tfair.enc_mass_rmse_kg(m), 'omitnan');
    idx = find(m, 1, 'first');
    pairMethods{i} = methodCol{idx};
end

fig = figure('Color', 'w', 'Position', [180, 180, 1050, 520]);
pairCats = categorical(pairs);
pairCats = reordercats(pairCats, pairs);
bar_qs_encoder(pairCats, M, payload_method_color(pairMethods, 'qs'), payload_method_color(pairMethods, 'encoder'));
ylabel('Mass RMSE [kg]');
title('QS vs Encoder mass error, common subset: load 2-4 kg, seed 42');
legend({'QS', 'Encoder'}, 'Location', 'northoutside', 'Orientation', 'horizontal');
grid on;
rotate_xticks(fig, 25);
finish_figure(fig, outDir, 'bar_qs_vs_encoder_mass_common_load2_4_seed42', saveOutputs);
end


function plot_heatmap_table(T, rowVar, colVar, valueVar, outDir, figName, saveOutputs)
print_data_sources(T, figName);
rowVals = table_text_column(T, rowVar);
colVals = table_text_column(T, colVar);
rows = unique(rowVals, 'stable');
cols = unique(colVals, 'stable');
M = NaN(numel(rows), numel(cols));
for i = 1:numel(rows)
    for j = 1:numel(cols)
        mask = strcmp(rowVals, rows{i}) & strcmp(colVals, cols{j});
        vals = T.(valueVar)(mask);
        if ~isempty(vals)
            M(i, j) = mean(vals, 'omitnan');
        end
    end
end
fig = figure('Color', 'w', 'Position', [180, 180, 760, 520]);
imagesc(M);
colorbar;
axis tight;
set(gca, 'XTick', 1:numel(cols), 'XTickLabel', cols, ...
    'YTick', 1:numel(rows), 'YTickLabel', rows, 'TickLabelInterpreter', 'none');
rotate_xticks(gcf, 30);
title(strrep(valueVar, '_', ' '));
for i = 1:numel(rows)
    for j = 1:numel(cols)
        if isfinite(M(i, j))
            text(j, i, sprintf('%.3g', M(i, j)), ...
                'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold');
        end
    end
end
finish_figure(fig, outDir, figName, saveOutputs);
end


function plot_one_timeseries(path, outDir, saveOutputs)
fprintf('\n[data source] timeseries\n  %s\n', path);
S = load(path);
[~, stem] = fileparts(path);
meta = parse_filename([stem, '.mat']);
encoderColor = payload_method_color(meta.method, 'encoder');
qsColor = payload_method_color(meta.method, 'qs');
dt = get_scalar(S, 'dt', 0.02);
ref = get_mat(S, 'payload_mass_ref_all');
if isempty(ref)
    return;
end
Tn = size(ref, 1);
t = (0:Tn - 1) * dt;
loadMask = get_mat(S, 'load_on_body_all') > 0.5;
if isempty(loadMask)
    loadMask = abs(ref) > 1.0e-6;
end
[~, envIdx] = max(sum(loadMask, 1));
envIdx = max(1, envIdx);

fig = figure('Color', 'w', 'Position', [100, 100, 1100, 680]);

subplot(3, 1, 1);
plot_line_if(S, t, 'payload_mass_ref_all', envIdx, 'true', 'k', 2.2); hold on;
plot_line_if(S, t, 'payload_mass_all', envIdx, 'QS', qsColor, 1.5, '--');
plot_line_if(S, t, 'encoder_mass_all', envIdx, 'Encoder', encoderColor, 1.8, '-');
ylabel('mass [kg]');
title(stem, 'Interpreter', 'none');
legend('Location', 'best');
grid on;

subplot(3, 1, 2);
plot_line_if(S, t, 'true_com_delta_x_all', envIdx, 'true', 'k', 2.2); hold on;
plot_line_if(S, t, 'modelC_com_delta_x_all', envIdx, 'QS', qsColor, 1.5, '--');
plot_line_if(S, t, 'encoder_com_delta_x_all', envIdx, 'Encoder', encoderColor, 1.8, '-');
ylabel('dCoM-x [m]');
legend('Location', 'best');
grid on;

subplot(3, 1, 3);
plot_line_if(S, t, 'true_com_delta_y_all', envIdx, 'true', 'k', 2.2); hold on;
plot_line_if(S, t, 'modelC_com_delta_y_all', envIdx, 'QS', qsColor, 1.5, '--');
plot_line_if(S, t, 'encoder_com_delta_y_all', envIdx, 'Encoder', encoderColor, 1.8, '-');
ylabel('dCoM-y [m]');
xlabel('time [s]');
legend('Location', 'best');
grid on;

finish_figure(fig, outDir, ['timeseries_', matlab.lang.makeValidName(stem)], saveOutputs);
end


function plot_line_if(S, t, field, envIdx, label, color, lw, style)
if nargin < 8 || isempty(style)
    style = '-';
end
A = get_mat(S, field);
if isempty(A)
    return;
end
if isvector(A)
    y = A(:);
else
    envIdx = min(envIdx, size(A, 2));
    y = A(:, envIdx);
end
if strcmp(style, '--')
    plot_dense_dash(t(:), y(:), label, color, lw);
else
    plot(t(:), y(:), style, 'DisplayName', label, 'Color', color, 'LineWidth', lw);
end
end


function plot_dense_dash(t, y, label, color, lw)
validT = t(isfinite(t));
if numel(validT) < 2
    plot(t, y, '-', 'DisplayName', label, 'Color', color, 'LineWidth', lw);
    return;
end
xRange = max(validT) - min(validT);
dtVals = diff(validT);
dtVals = dtVals(isfinite(dtVals) & dtVals > 0);
if isempty(dtVals)
    dt = max(xRange / 1000, eps);
else
    dt = median(dtVals);
end
dashLen = max(12 * dt, xRange / 45);
gapLen = max(3 * dt, xRange / 180);
phase = mod(t - validT(1), dashLen + gapLen);
yDash = y;
yDash(phase > dashLen) = NaN;
plot(t, yDash, '-', 'DisplayName', label, 'Color', color, 'LineWidth', lw);
end


function plot_one_scatter(path, outDir, saveOutputs)
fprintf('\n[data source] scatter\n  %s\n', path);
S = load(path);
[~, stem] = fileparts(path);
meta = parse_filename([stem, '.mat']);
encoderColor = payload_method_color(meta.method, 'encoder');
qsColor = payload_method_color(meta.method, 'qs');
ref = get_mat(S, 'payload_mass_ref_all');
qs = get_mat(S, 'payload_mass_all');
enc = get_mat(S, 'encoder_mass_all');
if isempty(ref) || isempty(qs)
    return;
end
mask = get_mat(S, 'load_on_body_all') > 0.5;
if isempty(mask)
    mask = abs(ref) > 1.0e-6;
end
validQs = mask & isfinite(ref) & isfinite(qs);
validEnc = ~isempty(enc) && isequal(size(enc), size(ref));

fig = figure('Color', 'w', 'Position', [150, 150, 900, 420]);

subplot(1, 2, 1);
scatter_downsample(ref(validQs), qs(validQs), 10, qsColor, false);
hold on; plot_identity(ref(validQs), qs(validQs));
xlabel('true mass [kg]');
ylabel('QS estimate [kg]');
title('QS');
grid on;

subplot(1, 2, 2);
if validEnc
    valid = mask & isfinite(ref) & isfinite(enc);
    scatter_downsample(ref(valid), enc(valid), 8, encoderColor, true);
    hold on; plot_identity(ref(valid), enc(valid));
end
xlabel('true mass [kg]');
ylabel('encoder estimate [kg]');
title('Encoder');
grid on;
annotation(fig, 'textbox', [0.05 0.94 0.9 0.05], 'String', stem, ...
    'Interpreter', 'none', 'EdgeColor', 'none', 'HorizontalAlignment', 'center');
finish_figure(fig, outDir, ['scatter_mass_', matlab.lang.makeValidName(stem)], saveOutputs);
end


function scatter_downsample(x, y, sz, color, filledMarker)
if nargin < 5 || isempty(filledMarker)
    filledMarker = true;
end
x = x(:); y = y(:);
n = numel(x);
if n > 8000
    idx = round(linspace(1, n, 8000));
    x = x(idx); y = y(idx);
end
if filledMarker
    scatter(x, y, sz, color, 'filled', 'MarkerFaceAlpha', 0.18, 'MarkerEdgeAlpha', 0.18);
else
    scatter(x, y, sz, 'MarkerEdgeColor', color, 'MarkerFaceColor', 'none', 'MarkerEdgeAlpha', 0.22);
end
end


function plot_identity(x, y)
vals = [x(:); y(:)];
vals = vals(isfinite(vals));
if isempty(vals)
    return;
end
lo = min(vals); hi = max(vals);
plot([lo hi], [lo hi], 'k--', 'LineWidth', 1.0);
axis equal;
xlim([lo hi]); ylim([lo hi]);
end


function c = table_text_column(T, varName)
v = T.(varName);
if iscell(v)
    c = v;
elseif iscategorical(v)
    c = cellstr(v);
elseif ischar(v)
    c = cellstr(v);
elseif isnumeric(v)
    c = cell(size(v, 1), 1);
    for i = 1:numel(v)
        c{i} = num2str(v(i));
    end
else
    try
        c = cellstr(v);
    catch
        c = repmat({''}, height(T), 1);
    end
end
c = c(:);
end


function print_data_sources(T, figName)
fprintf('\n[data source] %s\n', figName);
if isempty(T)
    fprintf('  <empty table>\n');
    return;
end
methodCol = table_text_column(T, 'method');
condCol = table_text_column(T, 'condition');
loadCol = table_text_column(T, 'load_range');
pathCol = table_text_column(T, 'mat_path');
for i = 1:height(T)
    fprintf('  %2d. method=%s | condition=%s | load=%s | seed=%g | %s\n', ...
        i, methodCol{i}, condCol{i}, loadCol{i}, T.seed(i), pathCol{i});
end
end


function rotate_xticks(fig, angleDeg)
axesList = findall(fig, 'Type', 'Axes');
for i = 1:numel(axesList)
    try
        axes(axesList(i)); %#ok<LAXES>
        xtickangle(angleDeg);
    catch
        % Older MATLAB versions may not have xtickangle. Leave labels as-is.
    end
end
end


function finish_figure(fig, outDir, name, saveOutputs)
if nargin < 4 || ~saveOutputs
    set(fig, 'Visible', 'on');
    drawnow;
    return;
end
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end
safeName = char(name);
pngPath = fullfile(outDir, [safeName, '.png']);
pdfPath = fullfile(outDir, [safeName, '.pdf']);
saveas(fig, pngPath);
try
    print(fig, pdfPath, '-dpdf', '-bestfit');
catch
    saveas(fig, pdfPath);
end
close(fig);
end
