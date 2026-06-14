% Plot the WF_TRON1A zero-joint-pose kinematic sketch from the URDF.
% Coordinate convention:
%   +X: forward, +Y: left, +Z: up
% Units: meters

clear; clc; close all;

%% Base collision box from robot.urdf
base_center = [0.030, 0.000, -0.072];
base_size   = [0.270, 0.260, 0.190];

%% Joint centers in base_Link frame, all joint angles are zero
L.abad  = [ 0.055560,  0.105000, -0.260200];
L.hip   = [-0.021440,  0.125500, -0.260200];
L.knee  = [-0.171440,  0.105000, -0.520010];
L.wheel = [-0.021440,  0.148500, -0.779820];

R.abad  = [ 0.055560, -0.105000, -0.260200];
R.hip   = [-0.021440, -0.125500, -0.260200];
R.knee  = [-0.171440, -0.105000, -0.520010];
R.wheel = [-0.021440, -0.148500, -0.779820];

%% Useful geometry
wheel_radius = 0.127;
wheel_width  = 0.050;

figure('Color', 'w', 'Name', 'WF_TRON1A zero pose structure');
hold on; grid on; axis equal;
view(42, 24);

plotBaseBox(base_center, base_size, [0.72 0.78 0.90], 0.25);
plotLeg(L, [0.05 0.32 0.95], 'L', wheel_radius, wheel_width);
plotLeg(R, [0.95 0.22 0.12], 'R', wheel_radius, wheel_width);
plotJointAxes(L, R);
plotLinkAngles(L, R);

% Draw the base frame axes.
axis_len = 0.18;
quiver3(0, 0, 0, axis_len, 0, 0, 'r', 'LineWidth', 2, ...
    'MaxHeadSize', 0.8, 'HandleVisibility', 'off');
quiver3(0, 0, 0, 0, axis_len, 0, 'g', 'LineWidth', 2, ...
    'MaxHeadSize', 0.8, 'HandleVisibility', 'off');
quiver3(0, 0, 0, 0, 0, axis_len, 'b', 'LineWidth', 2, ...
    'MaxHeadSize', 0.8, 'HandleVisibility', 'off');
text(axis_len, 0, 0, '+X forward', 'Color', 'r', ...
    'FontWeight', 'bold', 'HandleVisibility', 'off');
text(0, axis_len, 0, '+Y left', 'Color', 'g', ...
    'FontWeight', 'bold', 'HandleVisibility', 'off');
text(0, 0, axis_len, '+Z up', 'Color', 'b', ...
    'FontWeight', 'bold', 'HandleVisibility', 'off');

xlabel('X forward (m)');
ylabel('Y left (m)');
zlabel('Z up (m)');
title('WF\_TRON1A structure sketch, all joint angles = 0');

xlim([-0.35, 0.25]);
ylim([-0.28, 0.28]);
zlim([-0.90, 0.10]);

legend('show', 'Location', 'northeastoutside');

%% Print key drawing dimensions
fprintf('Zero-pose side-view points, left leg, [X Z] in meters:\n');
fprintf('  abad  = [% .6f, % .6f]\n', L.abad(1),  L.abad(3));
fprintf('  hip   = [% .6f, % .6f]\n', L.hip(1),   L.hip(3));
fprintf('  knee  = [% .6f, % .6f]\n', L.knee(1),  L.knee(3));
fprintf('  wheel = [% .6f, % .6f]\n\n', L.wheel(1), L.wheel(3));

printSegmentInfo('abad -> hip',  L.hip - L.abad);
printSegmentInfo('hip  -> knee', L.knee - L.hip);
printSegmentInfo('knee -> wheel', L.wheel - L.knee);
printLinkAngle('L hip angle, abad-hip-knee', L.abad - L.hip, L.knee - L.hip);
printLinkAngle('L knee angle, hip-knee-wheel', L.hip - L.knee, L.wheel - L.knee);

%% Local functions
function plotLeg(S, color, label_prefix, wheel_radius, wheel_width)
    pts = [S.abad; S.hip; S.knee; S.wheel];

    plot3(pts(:,1), pts(:,2), pts(:,3), '-', ...
        'Color', color, 'LineWidth', 3, 'DisplayName', [label_prefix ' leg']);
    scatter3(pts(:,1), pts(:,2), pts(:,3), 55, color, 'filled', ...
        'DisplayName', [label_prefix ' joints']);

    names = {'abad', 'hip', 'knee', 'wheel'};
    for i = 1:numel(names)
        text(pts(i,1), pts(i,2), pts(i,3), ['  ' label_prefix '_' names{i}], ...
            'Color', color, 'FontSize', 10, 'FontWeight', 'bold', ...
            'HandleVisibility', 'off');
    end

    plotWheel(S.wheel, wheel_radius, wheel_width, color, [label_prefix ' wheel']);
end

function plotWheel(center, radius, width, color, display_name)
    % Wheel axis is along Y in the URDF. Draw a thin cylinder along Y.
    n = 64;
    theta = linspace(0, 2*pi, n);
    y = [-width/2, width/2];
    [Theta, Y] = meshgrid(theta, y);
    X = center(1) + radius * cos(Theta);
    Z = center(3) + radius * sin(Theta);
    Y = center(2) + Y;

    surf(X, Y, Z, ...
        'FaceColor', color, 'FaceAlpha', 0.16, 'EdgeAlpha', 0.20, ...
        'DisplayName', display_name);

    plot3(center(1) + radius*cos(theta), center(2)*ones(size(theta)), ...
        center(3) + radius*sin(theta), '-', 'Color', color, 'LineWidth', 1.5, ...
        'HandleVisibility', 'off');
end

function plotJointAxes(L, R)
    % Joint axes from the URDF. All joint origins have rpy = 0 and all
    % joint angles are zero, so these axes are expressed in base_Link.
    axis_len = 0.095;
    axis_color = [0.05 0.05 0.05];

    axes_data = { ...
        L.abad, [1 0 0],  'L abad axis +X'; ...
        L.hip,  [0 1 0],  'L hip axis +Y'; ...
        L.knee, [0 -1 0], 'L knee axis -Y'; ...
        R.abad, [1 0 0],  'R abad axis +X'; ...
        R.hip,  [0 -1 0], 'R hip axis -Y'; ...
        R.knee, [0 1 0],  'R knee axis +Y'};

    for i = 1:size(axes_data, 1)
        p = axes_data{i, 1};
        a = axes_data{i, 2};
        name = axes_data{i, 3};
        a = a / norm(a);

        p0 = p - 0.5 * axis_len * a;
        p1 = p + 0.5 * axis_len * a;

        plot3([p0(1), p1(1)], [p0(2), p1(2)], [p0(3), p1(3)], ...
            '-', 'Color', axis_color, 'LineWidth', 2.0, ...
            'DisplayName', name);
        quiver3(p(1), p(2), p(3), ...
            0.5 * axis_len * a(1), ...
            0.5 * axis_len * a(2), ...
            0.5 * axis_len * a(3), ...
            0, 'Color', axis_color, 'LineWidth', 1.6, ...
            'MaxHeadSize', 0.9, 'HandleVisibility', 'off');
        text(p1(1), p1(2), p1(3), ['  ' name], ...
            'Color', axis_color, 'FontSize', 9, ...
            'HandleVisibility', 'off');
    end
end

function plotLinkAngles(L, R)
    angle_color = [0.35 0.05 0.55];

    drawAngleArc(L.hip, L.abad - L.hip, L.knee - L.hip, ...
        0.055, angle_color, 'L hip');
    drawAngleArc(L.knee, L.hip - L.knee, L.wheel - L.knee, ...
        0.065, angle_color, 'L knee');

    drawAngleArc(R.hip, R.abad - R.hip, R.knee - R.hip, ...
        0.055, angle_color, 'R hip');
    drawAngleArc(R.knee, R.hip - R.knee, R.wheel - R.knee, ...
        0.065, angle_color, 'R knee');
end

function drawAngleArc(center, v1, v2, radius, color, label_prefix)
    u1 = v1 / norm(v1);
    u2 = v2 / norm(v2);
    theta = acos(max(-1, min(1, dot(u1, u2))));

    % Build the shortest arc in the plane spanned by v1 and v2.
    n = 48;
    t = linspace(0, theta, n);
    normal = cross(u1, u2);
    if norm(normal) < 1e-10
        return;
    end
    normal = normal / norm(normal);
    basis2 = cross(normal, u1);

    arc = center + radius * (cos(t(:)) .* u1 + sin(t(:)) .* basis2);
    plot3(arc(:,1), arc(:,2), arc(:,3), '-', ...
        'Color', color, 'LineWidth', 2.2, ...
        'HandleVisibility', 'off');

    mid = center + 1.15 * radius * ...
        (cos(theta/2) * u1 + sin(theta/2) * basis2);
    text(mid(1), mid(2), mid(3), sprintf('  %s %.2f deg', label_prefix, rad2deg(theta)), ...
        'Color', color, 'FontSize', 9, 'FontWeight', 'bold', ...
        'HandleVisibility', 'off');
end

function plotBaseBox(center, size_vec, color, alpha_val)
    sx = size_vec(1) / 2;
    sy = size_vec(2) / 2;
    sz = size_vec(3) / 2;

    v = [ ...
        -sx -sy -sz;
         sx -sy -sz;
         sx  sy -sz;
        -sx  sy -sz;
        -sx -sy  sz;
         sx -sy  sz;
         sx  sy  sz;
        -sx  sy  sz] + center;

    f = [ ...
        1 2 3 4;
        5 6 7 8;
        1 2 6 5;
        2 3 7 6;
        3 4 8 7;
        4 1 5 8];

    patch('Vertices', v, 'Faces', f, ...
        'FaceColor', color, 'FaceAlpha', alpha_val, ...
        'EdgeColor', [0.25 0.30 0.40], 'LineWidth', 1.0, ...
        'DisplayName', 'base collision box');
end

function printSegmentInfo(name, v)
    len = norm(v);
    side_angle_from_down = atan2d(v(1), -v(3));
    fprintf('%s:\n', name);
    fprintf('  vector = [% .6f, % .6f, % .6f] m\n', v(1), v(2), v(3));
    fprintf('  length = %.9f m\n', len);
    fprintf('  side view X-Z angle from vertical down = %.9f deg\n\n', side_angle_from_down);
end

function printLinkAngle(name, v1, v2)
    angle_deg = acosd(max(-1, min(1, dot(v1, v2) / (norm(v1) * norm(v2)))));
    fprintf('%s = %.9f deg\n', name, angle_deg);
end
