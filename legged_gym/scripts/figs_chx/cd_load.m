function D = cd_load()
% 读取 co-design 画图数据 (由 prep_data.py 从 npz+BO日志 抽取)
here = fileparts(mfilename('fullpath'));
D = load(fullfile(here, 'data', 'codesign_figs_data.mat'));
end
