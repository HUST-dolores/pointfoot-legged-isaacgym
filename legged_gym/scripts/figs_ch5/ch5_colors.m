function c = ch5_colors()
% 第五章统一配色(约定:payload_method_color / Okabe-Ito 色盲友好色卡)。
% 顺序对应 variants = {Model-guided, Estimate-guided, Source-guided, RL-only}:
%   Model=E1 橙 #E69F00 | Estimate=E2 绿 #009E73 | Source=E3 天蓝 #56B4E9 | RL-only=E4 朱红 #D55E00
c = {[0.902 0.624 0.000], [0.000 0.620 0.451], [0.337 0.706 0.914], [0.835 0.369 0.000]};
% 顺带把中文字体设好(论文图标题用中文;Noto Sans CJK SC 也能正常渲染英文/变体名)
set(groot, 'defaultAxesFontName', 'Noto Sans CJK SC', 'defaultTextFontName', 'Noto Sans CJK SC');
end
