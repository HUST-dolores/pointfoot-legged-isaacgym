function plot_ig_actor_heatmap(outDir)
%PLOT_IG_ACTOR_HEATMAP Heatmap of actor IG (target='all', coarse) per input
% group, comparing E1-E5.
% rows = methods (E1, E2, E3, E5, E4)
% cols = input groups
% color = actor IG attribution %
% Each cell has numeric % overlay.
%
% Usage:
%   plot_ig_actor_heatmap()         % open figure, do NOT save
%   plot_ig_actor_heatmap('/dir')   % save PDF, keep figure open

if nargin < 1, outDir = ''; end
if ~isempty(outDir) && exist(outDir, 'dir') ~= 7, mkdir(outDir); end

ckpts = { ...
    'E1', 'exper_qs_resi_load_boost_3_seed_45_pemass'; ...
    'E2', 'exper_qs_noresi_load_boost_3_seed_45_pemass'; ...
    'E3', 'exper_history_only_load_boost_3_seed_45_pemass'; ...
    'E5', 'May25_01-00-47_exper_qs_resi_load_boost_3_no_torq_seed_45_pemass'; ...
    'E4', 'May24_19-30-38_exper_history_only_no_torq_load_boost_3_seed_45_pemass'; ...
};
% Display order: most-attended first (prev_actions), then torque-related groups,
% then encoder outputs (highlights est_mass < 0.5%).
groups = { ...
    'previous_actions',  'prev actions'; ...
    'dof_pos',           'dof pos'; ...
    'dof_vel',           'dof vel'; ...
    'projected_gravity', 'gravity'; ...
    'est_lin_vel',       'enc/est lin vel'; ...
    'torques',           'raw torques'; ...
    'qs_combined',       'QS features (load+resid)'; ...
    'est_com_delta',     'enc/est com'; ...
    'est_mass',          'enc/est mass'; ...
    'base_ang_vel',      'base ang vel'; ...
};
nC = size(ckpts, 1);
nG = size(groups, 1);
M = nan(nC, nG);

for c = 1:nC
    csvPath = find_latest_actor_ig(ckpts{c, 2});
    if isempty(csvPath)
        warning('No actor_ig CSV for %s', ckpts{c, 1}); continue;
    end
    attr = load_actor_coarse_all(csvPath);
    for g = 1:nG
        gName = groups{g, 1};
        if strcmp(gName, 'qs_combined')
            v = 0;
            if isKey(attr, 'qs_load_features'), v = v + attr('qs_load_features'); end
            if isKey(attr, 'qs_residual_baseline'), v = v + attr('qs_residual_baseline'); end
            M(c, g) = v;
        elseif isKey(attr, gName)
            M(c, g) = attr(gName);
        else
            M(c, g) = 0;
        end
    end
end

% --- Plot
fig = figure('Color', 'w', 'Position', [120, 120, 1100, 380]);
imagesc(M);
try, colormap(turbo); catch, colormap(parula); end
cb = colorbar;
cb.Label.String = 'Attribution [%]';
cb.Label.FontSize = 11;

set(gca, ...
    'XTick', 1:nG, 'XTickLabel', groups(:, 2), ...
    'YTick', 1:nC, 'YTickLabel', ckpts(:, 1), ...
    'TickLabelInterpreter', 'none', 'FontSize', 11);
try, xtickangle(25); catch, end
title('Actor IG attribution per input group (target = action, coarse)');
xlabel('Input group');
ylabel('Method');

cMax = max(M(:), [], 'omitnan');
if isempty(cMax) || ~isfinite(cMax) || cMax == 0, cMax = 1; end
for c = 1:nC
    for g = 1:nG
        v = M(c, g);
        if ~isfinite(v), continue; end
        ratio = v / cMax;
        if ratio > 0.55, txtColor = 'w'; else, txtColor = 'k'; end
        text(g, c, sprintf('%.1f', v), ...
             'HorizontalAlignment', 'center', ...
             'Color', txtColor, 'FontSize', 10, 'FontWeight', 'bold');
    end
end

if ~isempty(outDir)
    pdfPath = fullfile(outDir, 'ig_actor_heatmap.pdf');
    try
        exportgraphics(fig, pdfPath, 'ContentType', 'vector', 'BackgroundColor', 'white');
    catch
        print(fig, pdfPath, '-dpdf', '-bestfit');
    end
    fprintf('[plot_ig_actor_heatmap] saved %s\n', pdfPath);
end

fprintf('\nActor IG attribution (%%):\n');
fprintf('  %-6s ', 'ckpt');
for g = 1:nG, fprintf('%14s ', groups{g, 1}); end
fprintf('\n');
for c = 1:nC
    fprintf('  %-6s ', ckpts{c, 1});
    for g = 1:nG, fprintf('%14.2f ', M(c, g)); end
    fprintf('\n');
end
end


function attr = load_actor_coarse_all(csvPath)
attr = containers.Map('KeyType', 'char', 'ValueType', 'double');
fid = fopen(csvPath, 'r');
if fid < 0, return; end
header = strsplit(fgetl(fid), ',');
[~, iT]  = ismember('target', header);
[~, iGS] = ismember('groupset', header);
[~, iG]  = ismember('group', header);
[~, iP]  = ismember('percent', header);
while ~feof(fid)
    line = fgetl(fid);
    if ~ischar(line) || isempty(line), continue; end
    cols = strsplit(line, ',');
    if numel(cols) < max([iT, iGS, iG, iP]), continue; end
    if ~strcmp(cols{iT}, 'all'), continue; end
    if ~strcmp(cols{iGS}, 'coarse'), continue; end
    g = cols{iG};
    p = str2double(cols{iP});
    if isempty(g) || isnan(p), continue; end
    attr(g) = p;
end
fclose(fid);
end


function csvPath = find_latest_actor_ig(runName)
csvPath = '';
d = dir(fullfile(repo_root(), 'logs', 'wheelfoot_flat', 'WF_TRON1A', ...
                 runName, 'actor_ig', 'actor_ig_summary*.csv'));
if isempty(d), return; end
[~, idx] = max([d.datenum]);
csvPath = fullfile(d(idx).folder, d(idx).name);
end


function root = repo_root()
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
