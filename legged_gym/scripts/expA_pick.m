function r = expA_pick(R, variant, motion, cond)
% 从 run 数组 R 中挑出 (variant, motion, cond) 匹配的第一条；无则返回 []。
r = [];
for i = 1:numel(R)
    if strcmp(R(i).variant, variant) && strcmp(R(i).motion, motion) && strcmp(R(i).cond, cond)
        r = R(i); return;
    end
end
end
