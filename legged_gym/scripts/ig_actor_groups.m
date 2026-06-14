function ig_actor_groups(outDir)
%PLOT_IG_ACTOR_GROUPS Grouped-bar visualization of actor IG (target='all',
% coarse) for E1-E5. Each input group gets 5 colored bars (one per method).
% Supports paper conclusions J ("architecture marginal"), K ("est_mass unused"),
% L ("QS-in-obs ~6% stabilizing prior").
%
% x-axis: selected input groups (paper-relevant)
% y-axis: actor IG attribution [%]
% Bars: 5 method-colored bars per group, ordered E1, E2, E3, E5, E4
%
% Usage:
%   plot_ig_actor_groups()                 % open figure, do NOT save
%   plot_ig_actor_groups('/tmp/out')       % save PDF, close figure

if nargin < 1, outDir = ''; end
if ~isempty(outDir) && exist(outDir, 'dir') ~= 7, mkdir(outDir); end

ckpts = { ...
    'E1', 'exper_qs_resi_load_boost_3_seed_45_pemass'; ...
    'E2', 'exper_qs_noresi_load_boost_3_seed_45_pemass'; ...
    'E3', 'exper_history_only_load_boost_3_seed_45_pemass'; ...
    'E5', 'May25_01-00-47_exper_qs_resi_load_boost_3_no_torq_seed_45_pemass'; ...
    'E4', 'May24_19-30-38_exper_history_only_no_torq_load_boost_3_seed_45_pemass'; ...
};

% Groups to plot (in display order). 'qs_combined' is synthesized from
% qs_load_features + qs_residual_baseline. Groups absent in a ckpt show 0.
groups = { ...
    'previous_actions',  'previous actions'; ...
    'dof_pos',           'dof pos'; ...
    'est_lin_vel',       'enc/est lin vel'; ...
    'projected_gravity', 'projected gravity'; ...
    'dof_vel',           'dof vel'; ...
    'torques',           'raw torques'; ...
    'qs_combined',       'QS features combined'; ...
    'est_com_delta',     'enc/est com'; ...
    'est_mass',          'enc/est mass'; ...
    'base_ang_vel',      'base ang vel'; ...
};
nGroups = size(groups, 1);
nCkpts  = size(ckpts,  1);
M = zeros(nGroups, nCkpts);

for c = 1:nCkpts
    runName = ckpts{c, 2};
    csvPath = find_latest_actor_ig(runName);
    if isempty(csvPath)
        warning('No actor_ig_summary CSV found for %s (%s)', ckpts{c, 1}, runName);
        continue;
    end
    attr = load_actor_coarse_all(csvPath);
    for g = 1:nGroups
        gName = groups{g, 1};
        if strcmp(gName, 'qs_combined')
            v = 0;
            if isKey(attr, 'qs_load_features'), v = v + attr('qs_load_features'); end
            if isKey(attr, 'qs_residual_baseline'), v = v + attr('qs_residual_baseline'); end
            M(g, c) = v;
        elseif isKey(attr, gName)
            M(g, c) = attr(gName);
        end
    end
end

% --- Plot
fig = figure('Color', 'w', 'Position', [120, 120, 1100, 520]);
xCats = categorical(groups(:, 2));
xCats = reordercats(xCats, groups(:, 2));
b = bar(xCats, M);   % nGroups x nCkpts -> nCkpts bars per group
for k = 1:nCkpts
    b(k).FaceColor = payload_method_color(ckpts{k, 1});
    b(k).EdgeColor = 'none';
end
ylabel('Actor IG attribution [%] (target = full action)');
title('Actor IG per input group, E1-E5 (ckpt 11000, in-dist, coarse)');
legend(ckpts(:, 1), 'Location', 'northoutside', 'Orientation', 'horizontal');
grid on; set(gca, 'GridAlpha', 0.25);

% Rotate group labels for readability
try, xtickangle(25); catch, end

if ~isempty(outDir)
    pdfPath = fullfile(outDir, 'ig_actor_groups.pdf');
    try
        exportgraphics(fig, pdfPath, 'ContentType', 'vector', 'BackgroundColor', 'white');
    catch
        print(fig, pdfPath, '-dpdf', '-bestfit');
    end
    close(fig);
    fprintf('[plot_ig_actor_groups] saved %s\n', pdfPath);
end

% Print numerical table
fprintf('\nActor IG attribution per group (%%):\n');
fprintf('  %-22s ', 'group / ckpt');
for c = 1:nCkpts, fprintf('%7s ', ckpts{c, 1}); end
fprintf('\n');
for g = 1:nGroups
    fprintf('  %-22s ', groups{g, 1});
    for c = 1:nCkpts, fprintf('%7.2f ', M(g, c)); end
    fprintf('\n');
end
end


function attr = load_actor_coarse_all(csvPath)
% Returns Map: group_name -> percent  (target=='all', groupset=='coarse')
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
