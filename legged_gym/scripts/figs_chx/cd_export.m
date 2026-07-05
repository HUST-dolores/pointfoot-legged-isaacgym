function cd_export(f, name)
% 导出图到 figs_chx/out/ (PNG 位图 + PDF 矢量, 论文用)
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, 'out');
if ~exist(outdir, 'dir'), mkdir(outdir); end
set(f, 'PaperPositionMode', 'auto');
print(f, fullfile(outdir, [name '.png']), '-dpng', '-r200');
try
    exportgraphics(f, fullfile(outdir, [name '.pdf']), 'ContentType', 'vector');
catch
    print(f, fullfile(outdir, [name '.pdf']), '-dpdf', '-bestfit');  % 老版本 MATLAB 回退
end
fprintf('  -> %s.png / .pdf\n', name);
end
