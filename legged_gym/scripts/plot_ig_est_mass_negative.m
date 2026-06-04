function plot_ig_est_mass_negative(outDir)
%PLOT_IG_EST_MASS_NEGATIVE Single-panel bar plot of actor IG attribution to
% encoder/est_mass dim across E1-E5, with a reference line showing what
% attribution a uniform-attention actor would give. Directly visualizes
% paper conclusion K ("encoder mass head is essentially unused by actor").
%
% Reference baseline = 100 / actor_input_dim, computed per ckpt because the
% input dim differs (E1/E2 = 58 = 7 + 48 + 3; E3 = 46; E5 = 50; E4 = 38).
% A bar way below baseline = "actor specifically ignores this dim".
%
% Usage:
%   plot_ig_est_mass_negative()             % open figure, do NOT save
%   plot_ig_est_mass_negative('/tmp/out')   % save PDF, close figure

if nargin < 1, outDir = ''; end
if ~isempty(outDir) && exist(outDir, 'dir') ~= 7, mkdir(outDir); end

ckpts = { ...
    'E1', 'exper_qs_resi_load_boost_3_seed_45_pemass',              58; ...
    'E2', 'exper_qs_noresi_load_boost_3_seed_45_pemass',            58; ...
    'E3', 'exper_history_only_load_boost_3_seed_45_pemass',         46; ...
    'E5', 'May25_01-00-47_exper_qs_resi_load_boost_3_no_torq_seed_45_pemass', 50; ...
    'E4', 'May24_19-30-38_exper_history_only_no_torq_load_boost_3_seed_45_pemass', 38; ...
};
nCkpts = size(ckpts, 1);

estMass = zeros(nCkpts, 1);
uniform = zeros(nCkpts, 1);
for c = 1:nCkpts
    runName  = ckpts{c, 2};
    inputDim = ckpts{c, 3};
    csvPath = find_latest_actor_ig(runName);
    if isempty(csvPath)
        warning('No actor_ig for %s', ckpts{c, 1}); continue;
    end
    attr = load_actor_coarse_all(csvPath);
    if isKey(attr, 'est_mass'), estMass(c) = attr('est_mass'); end
    uniform(c) = 100 / inputDim;
end

% --- Plot
fig = figure('Color', 'w', 'Position', [120, 120, 720, 480]);
xCats = categorical(ckpts(:, 1));
xCats = reordercats(xCats, ckpts(:, 1));
b = bar(xCats, estMass, 'FaceColor', 'flat');
b.CData = payload_method_color(ckpts(:, 1));
b.EdgeColor = 'none';

hold on;
% Reference line: uniform attention baseline (per-dim share if attention were flat)
% Use a single representative line averaged across ckpts (they're within 1.5-2.6%)
yUniform = mean(uniform);
yline(yUniform, '--', sprintf('uniform baseline ≈ %.2f%%', yUniform), ...
      'LineWidth', 1.2, 'Color', [0.3 0.3 0.3], 'LabelHorizontalAlignment', 'right');

% Annotate bar values
for c = 1:nCkpts
    text(c, estMass(c) + 0.05, sprintf('%.2f%%', estMass(c)), ...
         'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

ylabel('Actor IG attribution to encoder/est\_mass [%]');
ylim([0, max(max(estMass)*2, yUniform * 1.3)]);
title({'Actor IG attribution to encoder mass output', ...
       'All 5 ckpts: < 0.5% (way below uniform baseline) \rightarrow effectively unused'});
grid on; set(gca, 'GridAlpha', 0.25);

if ~isempty(outDir)
    pdfPath = fullfile(outDir, 'ig_est_mass_negative.pdf');
    try
        exportgraphics(fig, pdfPath, 'ContentType', 'vector', 'BackgroundColor', 'white');
    catch
        print(fig, pdfPath, '-dpdf', '-bestfit');
    end
    close(fig);
    fprintf('[plot_ig_est_mass_negative] saved %s\n', pdfPath);
end

% Numerical table
fprintf('\nActor IG attribution to encoder/est_mass (%%):\n');
fprintf('  %-6s %-12s %-12s %s\n', 'ckpt', 'est_mass %', 'uniform %', 'ratio (est/uniform)');
for c = 1:nCkpts
    fprintf('  %-6s %-12.3f %-12.3f %.3f\n', ckpts{c, 1}, estMass(c), uniform(c), estMass(c) / uniform(c));
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
