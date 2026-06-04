function plot_ig_encoder_mass_heatmap(outDir)
%PLOT_IG_ENCODER_MASS_HEATMAP Heatmap of encoder mass-branch IG attribution.
% rows = methods (E1, E2, E3, E5, E4)
% cols = input groups (sorted by E1 attribution, descending)
% color = attribution %
% Each cell shows the numeric % overlaid in white/black depending on cell value.
%
% Usage:
%   plot_ig_encoder_mass_heatmap()         % open figure, do NOT save
%   plot_ig_encoder_mass_heatmap('/dir')   % save PDF, keep figure open

if nargin < 1, outDir = ''; end
if ~isempty(outDir) && exist(outDir, 'dir') ~= 7, mkdir(outDir); end

ckpts = { ...
    'E1', 'exper_qs_resi_load_boost_3_seed_45_pemass'; ...
    'E2', 'exper_qs_noresi_load_boost_3_seed_45_pemass'; ...
    'E3', 'exper_history_only_load_boost_3_seed_45_pemass'; ...
    'E5', 'May25_01-00-47_exper_qs_resi_load_boost_3_no_torq_seed_45_pemass'; ...
    'E4', 'May24_19-30-38_exper_history_only_no_torq_load_boost_3_seed_45_pemass'; ...
};
% Groups in display order — torque pathway (raw torques + QS) on the left,
% surrogate (prev_actions) middle, "other" (gravity/dof/etc) right.
groups = { ...
    'torques',              'raw torques'; ...
    'qs_load_features',     'QS load features'; ...
    'qs_residual_baseline', 'QS residual'; ...
    'previous_actions',     'prev actions'; ...
    'projected_gravity',    'gravity'; ...
    'dof_pos',              'dof pos'; ...
    'dof_vel',              'dof vel'; ...
    'base_ang_vel',         'base ang vel'; ...
};
nC = size(ckpts, 1);
nG = size(groups, 1);
M = nan(nC, nG);

for c = 1:nC
    csvPath = find_latest_encoder_ig(ckpts{c, 2});
    if isempty(csvPath)
        warning('No encoder_ig CSV for %s', ckpts{c, 1}); continue;
    end
    attr = aggregate_mass_attribution(csvPath);
    total = sum(values_of(attr));
    if total <= 0, continue; end
    for g = 1:nG
        v = 0;
        if isKey(attr, groups{g, 1}), v = attr(groups{g, 1}); end
        M(c, g) = 100 * v / total;
    end
end

% --- Plot
fig = figure('Color', 'w', 'Position', [120, 120, 1000, 380]);
imagesc(M);
% Use a perceptual colormap; if 'turbo' not available, fall back to 'parula'.
try, colormap(turbo); catch, colormap(parula); end
cb = colorbar;
cb.Label.String = 'Attribution [%]';
cb.Label.FontSize = 11;

set(gca, ...
    'XTick', 1:nG, 'XTickLabel', groups(:, 2), ...
    'YTick', 1:nC, 'YTickLabel', ckpts(:, 1), ...
    'TickLabelInterpreter', 'none', 'FontSize', 11);
try, xtickangle(25); catch, end
title('Encoder mass-branch IG attribution per source (% of total)');
xlabel('Input group');
ylabel('Method');

% Overlay numeric values on cells; switch text color based on cell intensity
cMax = max(M(:), [], 'omitnan');
if isempty(cMax) || ~isfinite(cMax) || cMax == 0, cMax = 1; end
for c = 1:nC
    for g = 1:nG
        v = M(c, g);
        if ~isfinite(v), continue; end
        % Choose text color for readability: white over dark cells, black over light
        ratio = v / cMax;
        if ratio > 0.55
            txtColor = 'w';
        else
            txtColor = 'k';
        end
        text(g, c, sprintf('%.1f', v), ...
             'HorizontalAlignment', 'center', ...
             'Color', txtColor, 'FontSize', 10, 'FontWeight', 'bold');
    end
end

if ~isempty(outDir)
    pdfPath = fullfile(outDir, 'ig_encoder_mass_heatmap.pdf');
    try
        exportgraphics(fig, pdfPath, 'ContentType', 'vector', 'BackgroundColor', 'white');
    catch
        print(fig, pdfPath, '-dpdf', '-bestfit');
    end
    fprintf('[plot_ig_encoder_mass_heatmap] saved %s\n', pdfPath);
end

% Print numerical table
fprintf('\nEncoder mass-branch attribution (%%):\n');
fprintf('  %-6s ', 'ckpt');
for g = 1:nG, fprintf('%14s ', groups{g, 1}); end
fprintf('\n');
for c = 1:nC
    fprintf('  %-6s ', ckpts{c, 1});
    for g = 1:nG, fprintf('%14.2f ', M(c, g)); end
    fprintf('\n');
end
end


function attr = aggregate_mass_attribution(csvPath)
attr = containers.Map('KeyType', 'char', 'ValueType', 'double');
fid = fopen(csvPath, 'r');
if fid < 0, return; end
header = strsplit(fgetl(fid), ',');
[~, iT] = ismember('target', header);
[~, iG] = ismember('group', header);
[~, iA] = ismember('attribution', header);
while ~feof(fid)
    line = fgetl(fid);
    if ~ischar(line) || isempty(line), continue; end
    cols = strsplit(line, ',');
    if numel(cols) < max([iT, iG, iA]), continue; end
    if ~strcmp(cols{iT}, 'mass'), continue; end
    g = cols{iG};
    if isempty(g) || startsWith(g, 'qs_dim_') || strcmp(g, 'extra_pre_action')
        continue;
    end
    a = str2double(cols{iA});
    if isnan(a), continue; end
    if isKey(attr, g), attr(g) = attr(g) + a; else, attr(g) = a; end
end
fclose(fid);
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
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
