function expA_savefig(fig, base)
% 存 PDF（矢量）+ PNG；老版本 MATLAB 回退到 saveas。
try
    exportgraphics(fig, [base '.pdf'], 'ContentType', 'vector');
    exportgraphics(fig, [base '.png'], 'Resolution', 150);
catch
    saveas(fig, [base '.png']);
end
fprintf('[expA] saved: %s.pdf/.png\n', base);
end
