function ig_encoder_window_heatmap(ckptLabel, targetName, nWindows, outDir)
%PLOT_IG_ENCODER_WINDOW_HEATMAP Rollout-time heatmap of encoder IG attribution.
%
% x-axis = rollout time, 1-second bins (matches --window_steps=50 @ 50 Hz)
% y-axis = input groups in fixed input order (no sorting by mean attribution)
% color = % attribution in that 1-s window
%
% The "raw" and "filtered" branches of the encoder input are merged
% (e.g., raw/torques + filtered/torques  ->  torques) so each variable
% has a single row.
%
% Usage:
%   plot_ig_encoder_window_heatmap                            % ALL 5 ckpts (E1/E2/E3/E5/E4) target=mass
%   plot_ig_encoder_window_heatmap('E5')                      % single ckpt E5, target=mass
%   plot_ig_encoder_window_heatmap('all', 'all')              % ALL 5 ckpts, target=full action
%   plot_ig_encoder_window_heatmap('E1', 'mass', 30)          % single ckpt cropped to first 30 windows
%   plot_ig_encoder_window_heatmap('E1', 'mass', inf, '/dir') % save PDF (no crop)
%
% Pass ckptLabel='' or 'all' to plot all 5 canonical ckpts (one figure each).
% Pass nWindows=Inf (or omit) to show all available windows. Underlying CSV
% is generated with --window_steps 50 --rollout_steps 2500 (50 windows / 50s).

if nargin < 1, ckptLabel = ''; end
if nargin < 2 || isempty(targetName),  targetName = 'mass'; end
if nargin < 3 || isempty(nWindows),    nWindows = inf; end
if nargin < 4, outDir = ''; end
if ~isempty(outDir) && exist(outDir, 'dir') ~= 7, mkdir(outDir); end

% Batch mode: ckptLabel empty or 'all' -> loop over all 5 canonical ckpts.
if isempty(ckptLabel) || strcmpi(ckptLabel, 'all')
    allCkpts = {'E1', 'E2', 'E3', 'E5', 'E4'};
    fprintf('[plot_ig_encoder_window_heatmap] batch mode: plotting %d ckpts (%s)\n', ...
            numel(allCkpts), strjoin(allCkpts, ', '));
    for i = 1:numel(allCkpts)
        plot_one(allCkpts{i}, targetName, nWindows, outDir);
    end
    return;
end

plot_one(ckptLabel, targetName, nWindows, outDir);
end


function one(ckptLabel, targetName, nWindows, outDir)

% Canonical ckpt mapping
ckptMap = containers.Map( ...
    {'E1','E2','E3','E5','E4'}, ...
    {'exper_qs_resi_load_boost_3_seed_45_pemass', ...
     'exper_qs_noresi_load_boost_3_seed_45_pemass', ...
     'exper_history_only_load_boost_3_seed_45_pemass', ...
     'May25_01-00-47_exper_qs_resi_load_boost_3_no_torq_seed_45_pemass', ...
     'May24_19-30-38_exper_history_only_no_torq_load_boost_3_seed_45_pemass'});
if ~isKey(ckptMap, ckptLabel)
    error('Unknown ckpt label "%s". Use one of: E1, E2, E3, E5, E4.', ckptLabel);
end
runName = ckptMap(ckptLabel);
csvPath = find_latest_window_csv(runName);
if isempty(csvPath)
    error(['No encoder_ig_windows_*.csv found for %s.\n', ...
           'Re-run encoder IG with --window_steps 50 to generate windowed data.'], ...
           ckptLabel);
end

% --- Read CSV
T = readtable(csvPath, 'Delimiter', ',');
T = T(strcmp(T.target, targetName), :);
if isempty(T)
    error('No rows for target "%s" in %s', targetName, csvPath);
end

% --- Merge raw/filtered: strip branch prefix from "label", recompute group key
%     e.g. "raw/torques" + "filtered/torques" -> "torques"
groupName = strings(height(T), 1);
for i = 1:height(T)
    lbl = char(T.label(i));
    slashIdx = strfind(lbl, '/');
    if isempty(slashIdx), groupName(i) = string(lbl);
    else,                 groupName(i) = string(lbl(slashIdx(1)+1:end));
    end
end
T.merged_group = groupName;

% Drop per-dim rows and overflow buckets
keep = ~startsWith(T.merged_group, 'qs_dim_') & ...
       T.merged_group ~= "extra_pre_action";
T = T(keep, :);

% --- Fixed Y-axis order, matches env.compute_group_observations + prev_actions:
%     ang_vel, gravity, dof_pos, dof_vel, [torques], [qs_load_features,
%     qs_residual_baseline], previous_actions.
yOrder = {'base_ang_vel', 'projected_gravity', 'dof_pos', 'dof_vel', ...
          'torques', 'qs_load_features', 'qs_residual_baseline', ...
          'previous_actions'};
present = false(size(yOrder));
for k = 1:numel(yOrder)
    present(k) = any(T.merged_group == string(yOrder{k}));
end
yLabels = yOrder(present);
nGroups = numel(yLabels);

% --- Unique windows + sort by start time
[winIds, ia] = unique(T.window_index, 'stable');
winTimes = T.time_start_s(ia);
[winTimes, sortIdx] = sort(winTimes);
winIds = winIds(sortIdx);

% Optional crop to first nWindows
if isfinite(nWindows) && nWindows > 0 && nWindows < numel(winIds)
    winIds = winIds(1:nWindows);
    winTimes = winTimes(1:nWindows);
end
nWin = numel(winIds);

% --- Build matrix M(nGroups, nWin) = sum of percent across raw + filtered
M = zeros(nGroups, nWin);
for w = 1:nWin
    wid = winIds(w);
    rowsW = T(T.window_index == wid, :);
    for g = 1:nGroups
        mask = rowsW.merged_group == string(yLabels{g});
        if any(mask), M(g, w) = sum(rowsW.percent(mask)); end
    end
end

% --- Plot heatmap
% Cascade window position based on how many figures are already open, so
% multiple calls produce visible separate windows instead of stacking.
existingFigs = findall(0, 'Type', 'figure');
nExisting = numel(existingFigs);
% Use a bigger step (80 px) and wrap to a 2nd column after 5 figures so it
% stays on screen.
xStep = 80; yStep = 80;
col = floor(nExisting / 5);   % move to next "column" every 5 figures
row = mod(nExisting, 5);
posX = 80 + col * 600 + row * xStep;
posY = 100 + row * yStep;
fig = figure('Color', 'w', ...
             'Position', [posX, posY, 1100, 380], ...
             'Name', sprintf('Encoder IG window heatmap (%s, %s)', ckptLabel, targetName));
set(fig, 'Visible', 'on');   % defensive: ensure GUI window comes up
figure(fig);                  % bring to front
drawnow;
fprintf('[plot_ig_encoder_window_heatmap] opened figure #%d at [%d,%d] (total open: %d)\n', ...
        fig.Number, posX, posY, nExisting + 1);
imagesc(M);
try, colormap(turbo); catch, colormap(parula); end

% Cap colormap at 30% so a dominant row (e.g., prev_actions = 60% in E5) doesn't
% squash the rest of the rows into a single dark shade. Values > 30% saturate
% but the numeric text overlay shows the true value.
clim([0, 30]);
cb = colorbar;
cb.Label.String = 'Attribution [%] (clipped at 30)';

% X tick: 1 label every ~5 windows to avoid clutter
ax = gca;
xtickStep = max(1, round(nWin / 10));
ax.XTick = 1:xtickStep:nWin;
ax.XTickLabel = arrayfun(@(t) sprintf('%.0fs', t), winTimes(1:xtickStep:nWin), ...
                         'UniformOutput', false);
ax.YTick = 1:nGroups;
ax.YTickLabel = yLabels;
ax.TickLabelInterpreter = 'none';
ax.FontSize = 11;

% Text overlay: only when cell width is wide enough to be readable (i.e. <=40 windows)
if nWin <= 40
    for r = 1:nGroups
        for c = 1:nWin
            v = M(r, c);
            if ~isfinite(v), continue; end
            if v / 30 > 0.55, txtColor = 'w'; else, txtColor = 'k'; end
            text(c, r, sprintf('%.0f', v), ...
                 'HorizontalAlignment', 'center', ...
                 'Color', txtColor, 'FontSize', 7);
        end
    end
end

xlabel('Rollout time (window start)');
ylabel('Input group (raw + filtered branches summed)');
title(sprintf('Encoder IG: %s | target=%s | window=1s (colormap clipped @ 30%%)', ckptLabel, targetName));

if ~isempty(outDir)
    if isfinite(nWindows)
        safeTag = sprintf('encoder_window_%s_%s_first%dwin', ckptLabel, targetName, nWin);
    else
        safeTag = sprintf('encoder_window_%s_%s', ckptLabel, targetName);
    end
    pdfPath = fullfile(outDir, [safeTag, '.pdf']);
    try
        exportgraphics(fig, pdfPath, 'ContentType', 'vector', 'BackgroundColor', 'white');
    catch
        print(fig, pdfPath, '-dpdf', '-bestfit');
    end
    fprintf('[plot_ig_encoder_window_heatmap] saved %s\n', pdfPath);
end

% Print summary stats
fprintf('\nEncoder IG window heatmap [%s, target=%s] %d windows x %d groups\n', ...
        ckptLabel, targetName, nWin, nGroups);
fprintf('  Row mean (%% over windows):\n');
for g = 1:nGroups
    fprintf('    %-22s %.2f%%\n', yLabels{g}, mean(M(g, :), 'omitnan'));
end
end


function csvPath = find_latest_window_csv(runName)
csvPath = '';
d = dir(fullfile(repo_root(), 'logs', 'wheelfoot_flat', 'WF_TRON1A', ...
                 runName, 'encoder_ig', 'encoder_ig_windows*.csv'));
if isempty(d), return; end
[~, idx] = max([d.datenum]);
csvPath = fullfile(d(idx).folder, d(idx).name);
end


function root = repo_root()
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
