function cd_export(f, name, export_pdf)
% 导出图到 figs_chx/out/。默认仅 PNG；明确传 true 时才导出 PDF。
if nargin < 3 || isempty(export_pdf); export_pdf = false; end
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, 'out');
if ~exist(outdir, 'dir'), mkdir(outdir); end
set(f, 'PaperPositionMode', 'auto');
drawnow;
try
    exportgraphics(f, fullfile(outdir, [name '.png']), ...
        'Resolution', 200, 'BackgroundColor', 'white');
catch
    print(f, fullfile(outdir, [name '.png']), '-dpng', '-r200');
end
if export_pdf
    try
        exportgraphics(f, fullfile(outdir, [name '.pdf']), 'ContentType', 'vector');
    catch
        print(f, fullfile(outdir, [name '.pdf']), '-dpdf', '-bestfit');  % 老版本 MATLAB 回退
    end
    fprintf('  -> %s.png / .pdf\n', name);
else
    fprintf('  -> %s.png\n', name);
end
end
