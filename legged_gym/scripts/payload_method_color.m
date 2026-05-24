function C = payload_method_color(method, variant)
%PAYLOAD_METHOD_COLOR Consistent method colors for payload experiment plots.
%
% Palette, in user-selected order:
%   1 #E69F00
%   2 #56B4E9
%   3 #009E73
%   4 #F0E442
%   5 #0072B2
%   6 #D55E00
%   7 #CC79A7
%
% Optional variant:
%   encoder/base  method color
%   qs            lighter version of the same method color

if nargin < 2 || isempty(variant)
    variant = 'base';
end
if iscell(method)
    C = zeros(numel(method), 3);
    for i = 1:numel(method)
        C(i, :) = payload_method_color(method{i}, variant);
    end
    return;
end

if isa(method, 'string')
    method = char(method);
end
method = char(method);

switch method
    case 'qs_residual'
        hex = '#E69F00';
    case 'history_only'
        hex = '#56B4E9';
    case 'qs_direct'
        hex = '#009E73';
    case 'qs_only'
        hex = '#F0E442';
    case 'oracle'
        hex = '#0072B2';
    case 'no_load_info'
        hex = '#D55E00';
    otherwise
        hex = '#CC79A7';
end

C = apply_variant(hex_to_rgb(hex), variant);
end


function rgb = hex_to_rgb(hex)
hex = char(hex);
if hex(1) == '#'
    hex = hex(2:end);
end
rgb = [hex2dec(hex(1:2)), hex2dec(hex(3:4)), hex2dec(hex(5:6))] / 255;
end


function C = apply_variant(baseColor, variant)
if isa(variant, 'string')
    variant = char(variant);
end
switch lower(char(variant))
    case {'base', 'encoder'}
        C = baseColor;
    case {'qs', 'analytic', 'baseline'}
        C = 0.55 * baseColor + 0.45 * [1 1 1];
    otherwise
        C = baseColor;
end
end
