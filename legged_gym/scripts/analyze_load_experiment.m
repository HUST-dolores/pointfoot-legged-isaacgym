% Analyze WF_TRON1A load estimation experiment.
% Put this script in the same folder as:
%   wf_deploy_log_real_load_fall_latest.csv
%
% Edit the event times below if you want to align the actual load steps
% more precisely with your video or notes.

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end

csvFile = fullfile(scriptDir, 'wf_deploy_log_real_load_fall_latest.csv');
T = readtable(csvFile);

t = T.t;
massEst = T.mass_est;
comx = T.comx;
comy = T.comy;
comz = T.comz;

% Actual payloads for the middle "add three loads" test.
% Assumption: "4,425 kg" means 4.425 kg. Change to 4.25 if needed.
loadMasses = [4.425, 3.48, 4.50];

% Approximate event times inferred from jumps in massEst.
% t = 135.4: first load appears
% t = 149.0: second load appears
% t = 158.1: third load appears
% t = 166.0, 169.8, 177.6: loads are removed in last-in first-out order
addTimes = [135.4, 149.0, 158.1];
removeTimes = [166.0, 169.8, 177.6];

% Later right-side load test markers, inferred from the same log.
rightAddTimes = [182.5, 192.1];
fallTime = 198.75;
massPlotXLim = [130.0, 180.0];
massEstOffset = 1.0;
bodyMass = 9.58;
bodyCom0 = [0.04576, 0.00014, -0.16398];

massEst = massEst + massEstOffset;

actualTimes = [min(t), addTimes, removeTimes, max(t)];
actualMass = [ ...
    0, ...
    loadMasses(1), ...
    loadMasses(1) + loadMasses(2), ...
    sum(loadMasses), ...
    loadMasses(1) + loadMasses(2), ...
    loadMasses(1), ...
    0, ...
    0 ...
];

massSmooth = movmedian(massEst, 51, 'omitnan');

%% Figure 1: estimated mass vs actual payload mass
figure('Name', 'Payload mass comparison', 'Color', 'w');
plot(t, massEst, 'Color', [0.75 0.75 0.75], ...
    'DisplayName', sprintf('estimated raw +%.1f kg', massEstOffset));
hold on;
plot(t, massSmooth, 'b', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('estimated movmedian +%.1f kg', massEstOffset));
stairs(actualTimes, actualMass, 'r', 'LineWidth', 2.0, 'DisplayName', 'actual payload');

for k = 1:numel(addTimes)
    if addTimes(k) >= massPlotXLim(1) && addTimes(k) <= massPlotXLim(2)
        xline(addTimes(k), '--k', sprintf('add %d', k), ...
            'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
    end
end
removeLabels = [3, 2, 1];
for k = 1:numel(removeTimes)
    if removeTimes(k) >= massPlotXLim(1) && removeTimes(k) <= massPlotXLim(2)
        xline(removeTimes(k), '--m', sprintf('remove %d', removeLabels(k)), ...
            'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
    end
end
if fallTime >= massPlotXLim(1) && fallTime <= massPlotXLim(2)
    xline(fallTime, '--r', 'fall', ...
        'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
end

grid on;
xlabel('time [s]');
ylabel('mass [kg]');
title('Estimated payload mass vs actual payload mass');
xlim(massPlotXLim);
legend('Location', 'south');

%% Print rough plateau errors.
fprintf('\nApproximate middle-load mass comparison\n');
fprintf('Assumed load masses: %.3f, %.3f, %.3f kg\n', loadMasses);
fprintf(['Add times: %.1f, %.1f, %.1f s; remove times, ', ...
    'last-in first-out: %.1f, %.1f, %.1f s\n\n'], ...
    addTimes, removeTimes);
fprintf('Applied estimated mass offset: +%.3f kg\n\n', massEstOffset);
fprintf('%8s %14s %14s %14s\n', 'window', 'actual_kg', 'est_med_kg', 'error_kg');

plateauStarts = [addTimes(1), addTimes(2), addTimes(3)];
plateauEnds = [addTimes(2), addTimes(3), removeTimes(1)];
actualPlateaus = cumsum(loadMasses);

for k = 1:numel(plateauStarts)
    % Use the middle of each plateau, avoiding the first two seconds of hand motion.
    t0 = plateauStarts(k) + 2.0;
    t1 = min(plateauEnds(k) - 1.0, t0 + 8.0);
    idx = t >= t0 & t <= t1;
    estMed = median(massEst(idx), 'omitnan');
    fprintf('%5.1f-%5.1f %14.3f %14.3f %14.3f\n', ...
        t0, t1, actualPlateaus(k), estMed, estMed - actualPlateaus(k));
end

%% Print whole loaded-period mean bias.
loadedStart = addTimes(1);
loadedEnd = removeTimes(end);
phaseStarts = [addTimes(1), addTimes(2), addTimes(3), removeTimes(1), removeTimes(2)];
phaseEnds = [addTimes(2), addTimes(3), removeTimes(1), removeTimes(2), removeTimes(3)];
phaseMasses = [ ...
    loadMasses(1), ...
    loadMasses(1) + loadMasses(2), ...
    sum(loadMasses), ...
    loadMasses(1) + loadMasses(2), ...
    loadMasses(1) ...
];
actualMean = sum((phaseEnds - phaseStarts) .* phaseMasses) / (loadedEnd - loadedStart);

meanTimes = [loadedStart; t(t > loadedStart & t < loadedEnd); loadedEnd];
massMeanValues = interp1(t, massEst, meanTimes, 'linear');
smoothMeanValues = interp1(t, massSmooth, meanTimes, 'linear');
rawMean = trapz(meanTimes, massMeanValues) / (loadedEnd - loadedStart);
smoothMean = trapz(meanTimes, smoothMeanValues) / (loadedEnd - loadedStart);

fprintf('\nWhole loaded-period mean bias\n');
fprintf('Loaded period: %.1f-%.1f s\n', loadedStart, loadedEnd);
fprintf('Actual red mean:          %8.3f kg\n', actualMean);
fprintf('Estimated raw mean:       %8.3f kg, residual %8.3f kg\n', ...
    rawMean, actualMean - rawMean);
fprintf('Estimated movmedian mean: %8.3f kg, residual %8.3f kg\n', ...
    smoothMean, actualMean - smoothMean);

%% Infer payload position from robot COM delta.
robotComDelta = [comx, comy, comz];
payloadPos = nan(size(robotComDelta));
validPayloadMass = massEst > 1e-6;
scale = (bodyMass + massEst(validPayloadMass)) ./ massEst(validPayloadMass);
payloadPos(validPayloadMass, :) = repmat(bodyCom0, sum(validPayloadMass), 1) ...
    + robotComDelta(validPayloadMass, :) .* repmat(scale, 1, 3);
loadedMask = t >= addTimes(1) & t < removeTimes(end);
payloadPosPlot = payloadPos;
payloadPosPlot(~loadedMask, :) = nan;

fprintf('\nPayload position inferred from robot COM delta\n');
fprintf(['Formula: payload_pos = body_com0 + ', ...
    '(body_mass + payload_mass) / payload_mass * robot_com_delta\n']);
fprintf('body_mass = %.3f kg, body_com0 = [%.5f %.5f %.5f]\n', ...
    bodyMass, bodyCom0);
fprintf('%8s %12s %12s %12s\n', 'window', 'load_x_m', 'load_y_m', 'load_z_m');
for k = 1:numel(plateauStarts)
    t0 = plateauStarts(k) + 2.0;
    t1 = min(plateauEnds(k) - 1.0, t0 + 8.0);
    idx = t >= t0 & t <= t1;
    med = median(payloadPos(idx, :), 1, 'omitnan');
    fprintf('%5.1f-%5.1f %12.3f %12.3f %12.3f\n', ...
        t0, t1, med(1), med(2), med(3));
end

%% Figure 2: estimated COM coordinates over time
figure('Name', 'Estimated COM coordinates', 'Color', 'w');
plot(t, comx, 'LineWidth', 1.2, 'DisplayName', 'comx');
hold on;
plot(t, comy, 'LineWidth', 1.2, 'DisplayName', 'comy');
plot(t, comz, 'LineWidth', 1.2, 'DisplayName', 'comz');

for k = 1:numel(addTimes)
    xline(addTimes(k), '--k', sprintf('add %d', k), ...
        'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
end
for k = 1:numel(removeTimes)
    xline(removeTimes(k), '--m', sprintf('remove %d', removeLabels(k)), ...
        'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
end
for k = 1:numel(rightAddTimes)
    xline(rightAddTimes(k), ':k', sprintf('right add %d', k), ...
        'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'top');
end
xline(fallTime, '--r', 'fall', ...
    'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');

grid on;
xlabel('time [s]');
ylabel('estimated robot COM offset [m]');
title('Estimated robot COM offset coordinates');
legend('Location', 'best');

%% Figure 3: COM x-y trajectory
figure('Name', 'Estimated COM xy trajectory', 'Color', 'w');
scatter(comx, comy, 8, t, 'filled');
hold on;
plot(comx, comy, 'Color', [0.7 0.7 0.7 0.35]);
plot(comx(1), comy(1), 'go', 'MarkerSize', 8, 'LineWidth', 1.5, ...
    'DisplayName', 'start');
[~, fallIdx] = min(abs(t - fallTime));
plot(comx(fallIdx), comy(fallIdx), 'rx', 'MarkerSize', 10, 'LineWidth', 2.0, ...
    'DisplayName', 'near fall');
grid on;
axis equal;
colorbar;
xlabel('robot COM delta x [m]');
ylabel('robot COM delta y [m]');
title('Estimated robot COM offset x-y trajectory, color = time [s]');
legend('Location', 'best');

%% Figure 4: payload position inferred from robot COM delta
figure('Name', 'Payload position inferred from robot COM', 'Color', 'w');
plot(t, payloadPosPlot(:, 1), 'LineWidth', 1.2, 'DisplayName', 'payload x inferred');
hold on;
plot(t, payloadPosPlot(:, 2), 'LineWidth', 1.2, 'DisplayName', 'payload y inferred');
plot(t, payloadPosPlot(:, 3), 'LineWidth', 1.2, 'DisplayName', 'payload z inferred');

for k = 1:numel(addTimes)
    if addTimes(k) >= massPlotXLim(1) && addTimes(k) <= massPlotXLim(2)
        xline(addTimes(k), '--k', sprintf('add %d', k), ...
            'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
    end
end
for k = 1:numel(removeTimes)
    if removeTimes(k) >= massPlotXLim(1) && removeTimes(k) <= massPlotXLim(2)
        xline(removeTimes(k), '--m', sprintf('remove %d', removeLabels(k)), ...
            'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
    end
end

grid on;
xlabel('time [s]');
ylabel('estimated payload position [m]');
title('Payload position inferred from robot COM offset');
xlim(massPlotXLim);
legend('Location', 'best');

%% Save figures for easy sharing.
saveas(findobj('Type', 'figure', 'Name', 'Payload mass comparison'), ...
    fullfile(scriptDir, 'payload_mass_comparison.png'));
saveas(findobj('Type', 'figure', 'Name', 'Estimated COM coordinates'), ...
    fullfile(scriptDir, 'payload_com_coordinates.png'));
saveas(findobj('Type', 'figure', 'Name', 'Estimated COM xy trajectory'), ...
    fullfile(scriptDir, 'payload_com_xy_trajectory.png'));
saveas(findobj('Type', 'figure', 'Name', 'Payload position inferred from robot COM'), ...
    fullfile(scriptDir, 'payload_position_from_robot_com.png'));

fprintf('\nSaved figures:\n');
fprintf('  %s\n', fullfile(scriptDir, 'payload_mass_comparison.png'));
fprintf('  %s\n', fullfile(scriptDir, 'payload_com_coordinates.png'));
fprintf('  %s\n', fullfile(scriptDir, 'payload_com_xy_trajectory.png'));
fprintf('  %s\n', fullfile(scriptDir, 'payload_position_from_robot_com.png'));
