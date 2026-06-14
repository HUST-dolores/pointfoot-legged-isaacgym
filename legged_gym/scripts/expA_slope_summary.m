function expA_slope_summary()
% 实验A — Figure：四变体转移曲线斜率汇总（真实质量 vs 向下力），分布内 [2,4]kg。
% 左=RL 编码器，右=QS(Model-C)。斜率越接近、力与质量越不可分；=1 表示完全当成等效质量。
% 与 expA_load_runs 一致：每段去前 1s 响应；拟合只用分布内 [2,4] 的点。
% 同时在控制台打印 per-env 统计（含力的 RMSE vs 力当量 / vs 0，全程 1–6kg）。

BAND = [2 4];
[R, EXPORT_DIR] = expA_load_runs();
variants = {'Model-guided', 'Source-guided', 'Estimate-guided', 'RL-only'};
sM = nan(4, 1); sF = nan(4, 1); iM = nan(4, 1); iF = nan(4, 1);
sMq = nan(4, 1); sFq = nan(4, 1);

for vi = 1:4
    rm = expA_pick(R, variants{vi}, 'walk', 'mass_sweep');
    rf = expA_pick(R, variants{vi}, 'walk', 'force_sweep');
    if ~isempty(rm); [sM(vi), iM(vi)] = fit_band(rm.ref, rm.rl, BAND); [sMq(vi), ~] = fit_band(rm.ref, rm.qs, BAND); end
    if ~isempty(rf); [sF(vi), iF(vi)] = fit_band(rf.ref, rf.rl, BAND); [sFq(vi), ~] = fit_band(rf.ref, rf.qs, BAND); end
end

fig = figure('Color', 'w', 'Position', [100 100 900 420], 'Name', 'Exp A 斜率汇总');
subplot(1, 2, 1);
bar([sM sF]); set(gca, 'XTickLabel', variants, 'XTickLabelRotation', 20);
yline(1, '--', 'Color', [.6 .6 .6]); grid on; ylim([0 1.3]);
legend({'真实质量', '向下力'}, 'Location', 'south'); ylabel('转移曲线斜率'); title('RL 编码器');
subplot(1, 2, 2);
bar([sMq sFq]); set(gca, 'XTickLabel', variants, 'XTickLabelRotation', 20);
yline(1, '--', 'Color', [.6 .6 .6]); grid on; ylim([0 1.3]);
legend({'真实质量', '向下力'}, 'Location', 'south'); ylabel('转移曲线斜率'); title('QS (Model-C)');
sgtitle('斜率对比（分布内 [2,4]，每段去前1s）：越接近越不可分（=1 表示力被完全当成等效质量）', 'FontWeight', 'bold');
expA_savefig(fig, fullfile(EXPORT_DIR, 'expA_slope_summary'));

% ---- 控制台 per-env 统计 ----
fprintf('\n==== per-env 转移曲线（行走, RL 编码器；斜率=分布内[2,4]，RMSE=全程1-6）====\n');
fprintf('%-16s | 质量[2,4]:斜率 截距 | 力[2,4]:斜率 截距 | 力RMSE(vs力当量) | 力RMSE(vs0)\n', '变体');
for vi = 1:4
    rf = expA_pick(R, variants{vi}, 'walk', 'force_sweep');
    s = sprintf('%-16s |', variants{vi});
    s = [s sprintf(' %5.2f %+5.2f |', sM(vi), iM(vi))];
    if ~isempty(rf)
        rmse_feq = sqrt(mean((rf.rl(:) - rf.ref(:)).^2, 'omitnan'));
        rmse_0   = sqrt(mean((rf.rl(:)).^2, 'omitnan'));
        s = [s sprintf(' %5.2f %+5.2f | %8.3f kg | %7.3f kg', sF(vi), iF(vi), rmse_feq, rmse_0)];
    else
        s = [s '   --        |       --       |     --'];
    end
    fprintf('%s\n', s);
end
end

% ------------------------------------------------------------------
function [a, b] = fit_band(x, y, band)
x = x(:); y = y(:);
sel = x >= band(1) & x <= band(2) & isfinite(x) & isfinite(y);
[a, b] = expA_linfit(x(sel), y(sel));
end
