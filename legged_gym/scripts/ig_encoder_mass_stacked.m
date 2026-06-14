function ig_encoder_mass_stacked(outDir)
%PLOT_IG_ENCODER_MASS_STACKED Stacked-bar visualization of encoder mass-branch
% IG attribution, comparing the four paper methods. Directly visualizes paper contribution (b):
% the "two-pathway torque signal" mechanism (torques + qs_features = ~54% in
% Model-guided RL, ~20% in RL + model output, ~27% in RL + model input,
% 0% in RL-only).
%
% x-axis: 4 methods ordered Model-guided RL, RL + model input,
% RL + model output, RL-only
% y-axis: % of total mass-branch attribution
% Stack layers (bottom-up):
%   model input (torques; only present when use_torques_in_obs=True)
%   model output  (QS load-estimation output + QS residual baseline)
%   previous_actions   (the "implicit torque surrogate" pathway)
%   other (everything else)
%
% Usage from MATLAB:
%   plot_ig_encoder_mass_stacked()                 % open figure AND save PDF
%   plot_ig_encoder_mass_stacked('/tmp/myout')     % open figure AND save PDF to dir
%
% Figure stays open (not closed after save). Default save dir is
% logs/wheelfoot_flat/WF_TRON1A/exported/ig_pdfs/.

if nargin < 1 || isempty(outDir)
    outDir = fullfile(repo_root(), 'logs', 'wheelfoot_flat', 'WF_TRON1A', 'exported', 'ig_pdfs');
end
if exist(outDir, 'dir') ~= 7, mkdir(outDir); end

% Canonical ckpts: code -> run directory -> display label.
ckpts = { ...
    'E1', 'exper_qs_resi_load_boost_3_seed_45_pemass', ...
          'Model-guided RL'; ...
    'E3', 'exper_history_only_load_boost_3_seed_45_pemass', ...
          'RL + model input'; ...
    'E5', 'May25_01-00-47_exper_qs_resi_load_boost_3_no_torq_seed_45_pemass', ...
          'RL + model output'; ...
    'E4', 'May24_19-30-38_exper_history_only_no_torq_load_boost_3_seed_45_pemass', ...
          'RL-only'; ...
};

% Stacked layers (bottom-up; semantic grouping for contribution-(b) narrative)
layers = { ...
    'torques',                                      'model input';      % direct torque pathway
    {'qs_load_features', 'qs_residual_baseline'},   'model output';     % QS/model output + residual baseline
    'previous_actions',                             'previous actions'; % implicit surrogate
    'other',                                        'other';
};
nLayers = size(layers, 1);
nCkpts = size(ckpts, 1);
M = zeros(nLayers, nCkpts);     % stack values [%]

for c = 1:nCkpts
    label   = ckpts{c, 1};
    runName = ckpts{c, 2};
    csvPath = find_latest_encoder_ig(runName);
    if isempty(csvPath)
        warning('No encoder_ig_summary CSV found for %s (%s)', label, runName);
        continue;
    end
    attrByGroup = aggregate_mass_attribution(csvPath);
    total = sum(values_of(attrByGroup));
    if total <= 0, continue; end
    used = 0;
    for k = 1:nLayers - 1   % all explicit non-"other" layers
        v = sum_layer(attrByGroup, layers{k, 1});
        M(k, c) = 100 * v / total;
        used = used + v;
    end
    % "other" = everything not explicitly stacked above
    M(nLayers, c) = 100 * (total - used) / total;
end

% --- Plot
% Colors: bottom layers (model inputs + model outputs) use a warm gradient to emphasize
% the "torque-derived pathway". Surrogate (prev_actions) + other use cool.
layerColors = [ ...
    0.86, 0.20, 0.18; ...  % model input    deep red
    0.95, 0.55, 0.20; ...  % model output   orange
    0.30, 0.55, 0.85; ...  % prev_actions   steel blue
    0.65, 0.65, 0.65; ...  % other          grey
];

fig = figure('Color', 'w', 'Position', [120, 120, 760, 520]);
displayLabels = ckpts(:, 3);
xCats = categorical(displayLabels);
xCats = reordercats(xCats, displayLabels);   % preserve paper-method order
b = bar(xCats, M', 'stacked');
for k = 1:nLayers
    b(k).FaceColor = layerColors(k, :);
    b(k).EdgeColor = 'w';
    b(k).LineWidth = 0.6;
end

% Directly label each visible stack segment inside every bar.
segmentLabels = { ...
    sprintf('model\ninput'); ...
    sprintf('model\noutput'); ...
    sprintf('previous\nactions'); ...
    'other'; ...
};
segmentTextColors = [ ...
    1.00, 1.00, 1.00; ...
    0.10, 0.10, 0.10; ...
    1.00, 1.00, 1.00; ...
    0.10, 0.10, 0.10; ...
];
for c = 1:nCkpts
    for k = 1:nLayers
        v = M(k, c);
        if v < 4.0
            continue;
        end
        yBase = sum(M(1:k-1, c));
        if v < 8.0
            fontSize = 6;
        else
            fontSize = 8;
        end
        text(c, yBase + 0.5 * v, segmentLabels{k}, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
             'FontSize', fontSize, 'FontWeight', 'bold', ...
             'Color', segmentTextColors(k, :));
    end
end

ylabel('Normalized IG attribution [%]');
ylim([0 100]);
title('Input Attribution of the RL-based Mass Estimator');
grid on;
set(gca, 'GridAlpha', 0.25);

% Save PDF but keep figure open for inspection.
pdfPath = fullfile(outDir, 'ig_encoder_mass_stacked.pdf');
try
    exportgraphics(fig, pdfPath, 'ContentType', 'vector', 'BackgroundColor', 'white');
catch
    print(fig, pdfPath, '-dpdf', '-bestfit');
end
fprintf('[plot_ig_encoder_mass_stacked] saved %s\n', pdfPath);

% Print the numerical table for the figure
fprintf('\nEncoder mass-branch attribution (%%):\n');
fprintf('  %-25s ', 'group / ckpt');
for c = 1:nCkpts, fprintf('%20s ', ckpts{c, 3}); end
fprintf('\n');
for k = 1:nLayers
    fprintf('  %-25s ', layers{k, 2});
    for c = 1:nCkpts, fprintf('%20.2f ', M(k, c)); end
    fprintf('\n');
end
end


function attr = aggregate_mass_attribution(csvPath)
%AGGREGATE_MASS_ATTRIBUTION  Sum attribution per group for target='mass',
% combining raw + filtered branches; skip per-dim 'qs_dim_*' entries.
attr = containers.Map('KeyType', 'char', 'ValueType', 'double');
fid = fopen(csvPath, 'r');
if fid < 0, return; end
header = strsplit(fgetl(fid), ',');
[~, idxTarget] = ismember('target', header);
[~, idxGroup]  = ismember('group', header);
[~, idxAttr]   = ismember('attribution', header);
while ~feof(fid)
    line = fgetl(fid);
    if ~ischar(line) || isempty(line), continue; end
    cols = strsplit(line, ',');
    if numel(cols) < max([idxTarget, idxGroup, idxAttr]), continue; end
    if ~strcmp(cols{idxTarget}, 'mass'), continue; end
    g = cols{idxGroup};
    if isempty(g) || startsWith(g, 'qs_dim_') || strcmp(g, 'extra_pre_action')
        continue;
    end
    a = str2double(cols{idxAttr});
    if isnan(a), continue; end
    if isKey(attr, g)
        attr(g) = attr(g) + a;
    else
        attr(g) = a;
    end
end
fclose(fid);
end


function v = get_or_zero(map, key)
if isKey(map, key), v = map(key); else, v = 0; end
end


function v = sum_layer(map, layerKeys)
if ischar(layerKeys) || isstring(layerKeys)
    v = get_or_zero(map, char(layerKeys));
    return;
end
v = 0;
for i = 1:numel(layerKeys)
    v = v + get_or_zero(map, layerKeys{i});
end
end


function vals = values_of(map)
keysList = keys(map);
vals = zeros(numel(keysList), 1);
for i = 1:numel(keysList), vals(i) = map(keysList{i}); end
end


function csvPath = find_latest_encoder_ig(runName)
csvPath = '';
d = dir(fullfile(repo_root(), 'logs', 'wheelfoot_flat', 'WF_TRON1A', ...
                 runName, 'encoder_ig', 'encoder_ig_summary*.csv'));
if isempty(d), return; end
[~, idx] = max([d.datenum]);
csvPath = fullfile(d(idx).folder, d(idx).name);
end


function root = repo_root()
% This file lives in legged_gym/scripts/; repo root is 2 levels up.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
