function [a, b] = expA_linfit(x, y)
% 一次多项式拟合 y = a*x + b（剔除 NaN）。per-env 点直接进，不做跨环境平均。
x = x(:); y = y(:); ok = isfinite(x) & isfinite(y);
if nnz(ok) < 2; a = NaN; b = NaN; return; end
p = polyfit(x(ok), y(ok), 1); a = p(1); b = p(2);
end
