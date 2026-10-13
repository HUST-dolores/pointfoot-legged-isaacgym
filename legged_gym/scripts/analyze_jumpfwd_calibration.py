"""Analyze fixed-command jump grids and build a 2-D command calibration LUT."""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def load_rows(paths):
    rows = []
    for source_id, path in enumerate(paths):
        with open(path, newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                parsed = {key: float(value) for key, value in row.items()}
                parsed["source_id"] = source_id
                parsed["source_path"] = path
                rows.append(parsed)
    return rows


def percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def summarize_points(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(round(row["hstar"], 7), round(row["dstar"], 7))].append(row)

    summaries = []
    for (hstar, dstar), group in sorted(groups.items()):
        successful = [row for row in group if int(row["takeoffs"]) == 1]
        height = np.asarray([row["height"] for row in successful])
        distance = np.asarray([row["disp"] for row in successful])
        lateral = np.asarray([row["land_lateral_disp"] for row in successful])
        max_roll = np.asarray([row["max_roll"] for row in successful])

        repeat_groups = defaultdict(list)
        for row in successful:
            repeat_groups[(int(row["source_id"]), int(row["env_id"]))].append(row)
        repeat_height_std = []
        repeat_distance_std = []
        for repeat_group in repeat_groups.values():
            if len(repeat_group) >= 2:
                repeat_height_std.append(np.std([row["height"] for row in repeat_group], ddof=1))
                repeat_distance_std.append(np.std([row["disp"] for row in repeat_group], ddof=1))

        summaries.append({
            "hstar": hstar,
            "dstar": dstar,
            "n": len(group),
            "single_jump_n": len(successful),
            "single_jump_rate": len(successful) / max(len(group), 1),
            "height_mean": float(np.mean(height)) if len(height) else float("nan"),
            "height_std": float(np.std(height, ddof=1)) if len(height) > 1 else 0.0,
            "height_p05": percentile(height, 5),
            "height_p95": percentile(height, 95),
            "distance_mean": float(np.mean(distance)) if len(distance) else float("nan"),
            "distance_std": float(np.std(distance, ddof=1)) if len(distance) > 1 else 0.0,
            "distance_p05": percentile(distance, 5),
            "distance_p95": percentile(distance, 95),
            "repeat_height_std_median": float(np.median(repeat_height_std)) if repeat_height_std else float("nan"),
            "repeat_height_std_p95": percentile(repeat_height_std, 95),
            "repeat_distance_std_median": float(np.median(repeat_distance_std)) if repeat_distance_std else float("nan"),
            "repeat_distance_std_p95": percentile(repeat_distance_std, 95),
            "lateral_abs_mean": float(np.mean(np.abs(lateral))) if len(lateral) else float("nan"),
            "lateral_abs_p95": percentile(np.abs(lateral), 95),
            "max_roll_deg_p95": percentile(np.degrees(max_roll), 95),
        })
    return summaries


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def point_arrays(summaries):
    valid = [
        row for row in summaries
        if row["single_jump_n"] > 0 and np.isfinite(row["height_mean"]) and np.isfinite(row["distance_mean"])
    ]
    commands = np.asarray([[row["hstar"], row["dstar"]] for row in valid])
    actual = np.asarray([[row["height_mean"], row["distance_mean"]] for row in valid])
    return valid, commands, actual


def fit_affine(commands, actual):
    design = np.column_stack((np.ones(len(commands)), commands))
    coefficients, _, _, _ = np.linalg.lstsq(design, actual, rcond=None)
    predicted = design @ coefficients
    residual = actual - predicted
    total = actual - actual.mean(axis=0)
    r2 = 1.0 - np.sum(residual ** 2, axis=0) / np.maximum(np.sum(total ** 2, axis=0), 1e-12)
    linear = coefficients[1:, :].T
    return coefficients, {
        "r2_height": float(r2[0]),
        "r2_distance": float(r2[1]),
        "rmse_height": float(np.sqrt(np.mean(residual[:, 0] ** 2))),
        "rmse_distance": float(np.sqrt(np.mean(residual[:, 1] ** 2))),
        "linear_matrix": linear.tolist(),
        "offset": coefficients[0, :].tolist(),
        "determinant": float(np.linalg.det(linear)),
        "condition_number": float(np.linalg.cond(linear)),
    }


class RegularSurface:
    def __init__(self, summaries):
        self.heights = np.asarray(sorted({row["hstar"] for row in summaries}))
        self.distances = np.asarray(sorted({row["dstar"] for row in summaries}))
        self.values = np.full((len(self.heights), len(self.distances), 2), np.nan)
        for row in summaries:
            h_index = int(np.where(np.isclose(self.heights, row["hstar"]))[0][0])
            d_index = int(np.where(np.isclose(self.distances, row["dstar"]))[0][0])
            self.values[h_index, d_index] = (row["height_mean"], row["distance_mean"])
        if np.isnan(self.values).any():
            raise ValueError("Calibration input must contain a successful result at every regular-grid point")

    @staticmethod
    def _bracket(axis, value):
        value = float(np.clip(value, axis[0], axis[-1]))
        upper = min(max(int(np.searchsorted(axis, value, side="right")), 1), len(axis) - 1)
        lower = upper - 1
        weight = (value - axis[lower]) / max(axis[upper] - axis[lower], 1e-12)
        return lower, upper, weight

    def evaluate(self, height, distance):
        h0, h1, wh = self._bracket(self.heights, height)
        d0, d1, wd = self._bracket(self.distances, distance)
        low = (1.0 - wd) * self.values[h0, d0] + wd * self.values[h0, d1]
        high = (1.0 - wd) * self.values[h1, d0] + wd * self.values[h1, d1]
        return (1.0 - wh) * low + wh * high

    def dense_lut(self, samples_per_axis):
        dense_h = np.linspace(self.heights[0], self.heights[-1], samples_per_axis)
        dense_d = np.linspace(self.distances[0], self.distances[-1], samples_per_axis)
        rows = []
        for height in dense_h:
            for distance in dense_d:
                actual = self.evaluate(height, distance)
                rows.append({
                    "actual_height": float(actual[0]),
                    "actual_distance": float(actual[1]),
                    "command_height": float(height),
                    "command_distance": float(distance),
                })
        return rows

    def cell_jacobians(self):
        diagnostics = []
        for h_index in range(len(self.heights) - 1):
            for d_index in range(len(self.distances) - 1):
                dh = self.heights[h_index + 1] - self.heights[h_index]
                dd = self.distances[d_index + 1] - self.distances[d_index]
                left = 0.5 * (self.values[h_index, d_index] + self.values[h_index, d_index + 1])
                right = 0.5 * (self.values[h_index + 1, d_index] + self.values[h_index + 1, d_index + 1])
                low = 0.5 * (self.values[h_index, d_index] + self.values[h_index + 1, d_index])
                high = 0.5 * (self.values[h_index, d_index + 1] + self.values[h_index + 1, d_index + 1])
                jacobian = np.column_stack(((right - left) / dh, (high - low) / dd))
                diagnostics.append({
                    "command_height": float(0.5 * (self.heights[h_index] + self.heights[h_index + 1])),
                    "command_distance": float(0.5 * (self.distances[d_index] + self.distances[d_index + 1])),
                    "determinant": float(np.linalg.det(jacobian)),
                    "condition_number": float(np.linalg.cond(jacobian)),
                    "dh_actual_dh_cmd": float(jacobian[0, 0]),
                    "dh_actual_dd_cmd": float(jacobian[0, 1]),
                    "dd_actual_dh_cmd": float(jacobian[1, 0]),
                    "dd_actual_dd_cmd": float(jacobian[1, 1]),
                })
        return diagnostics


def convex_hull(points):
    unique = sorted(set(map(tuple, points)))
    if len(unique) <= 1:
        return [list(point) for point in unique]

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return [list(point) for point in lower[:-1] + upper[:-1]]


def validate_heldout(surface, inverse_lut, heldout_rows):
    summaries = summarize_points(heldout_rows)
    _, commands, actual = point_arrays(summaries)
    predicted = np.asarray([surface.evaluate(*command) for command in commands])
    output_span = np.maximum(np.ptp(inverse_lut[:, :2], axis=0), 1e-6)
    validation = []
    inverse_commands = []
    for command, target, estimate in zip(commands, actual, predicted):
        normalized_error = (inverse_lut[:, :2] - target) / output_span
        nearest = int(np.argmin(np.sum(normalized_error ** 2, axis=1)))
        inverse_command = inverse_lut[nearest, 2:]
        inverse_commands.append(inverse_command)
        validation.append({
            "command_height": float(command[0]),
            "command_distance": float(command[1]),
            "actual_height": float(target[0]),
            "actual_distance": float(target[1]),
            "predicted_height": float(estimate[0]),
            "predicted_distance": float(estimate[1]),
            "inverse_command_height": float(inverse_command[0]),
            "inverse_command_distance": float(inverse_command[1]),
        })
    inverse_commands = np.asarray(inverse_commands)
    error = predicted - actual
    inverse_error = inverse_commands - commands
    metrics = {
        "point_count": len(commands),
        "forward_rmse_height": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "forward_rmse_distance": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "forward_mae_height": float(np.mean(np.abs(error[:, 0]))),
        "forward_mae_distance": float(np.mean(np.abs(error[:, 1]))),
        "inverse_command_mae_height": float(np.mean(np.abs(inverse_error[:, 0]))),
        "inverse_command_mae_distance": float(np.mean(np.abs(inverse_error[:, 1]))),
    }
    return validation, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--heldout", nargs="*", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lut-samples", type=int, default=81)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_rows(args.inputs)
    summaries = summarize_points(rows)
    write_csv(os.path.join(args.output_dir, "point_statistics.csv"), summaries)
    _, commands, actual = point_arrays(summaries)
    _, affine_metrics = fit_affine(commands, actual)
    surface = RegularSurface(summaries)
    lut_rows = surface.dense_lut(args.lut_samples)
    write_csv(os.path.join(args.output_dir, "inverse_lut.csv"), lut_rows)
    jacobians = surface.cell_jacobians()
    write_csv(os.path.join(args.output_dir, "cell_jacobians.csv"), jacobians)

    success_rates = np.asarray([row["single_jump_rate"] for row in summaries])
    repeat_height = np.asarray([row["repeat_height_std_median"] for row in summaries])
    repeat_distance = np.asarray([row["repeat_distance_std_median"] for row in summaries])
    determinants = np.asarray([row["determinant"] for row in jacobians])
    conditions = np.asarray([row["condition_number"] for row in jacobians])
    report = {
        "input_files": args.inputs,
        "jump_count": len(rows),
        "grid_point_count": len(summaries),
        "single_jump_rate_min": float(np.min(success_rates)),
        "single_jump_rate_mean": float(np.mean(success_rates)),
        "repeat_height_std_median_across_points": float(np.nanmedian(repeat_height)),
        "repeat_distance_std_median_across_points": float(np.nanmedian(repeat_distance)),
        "actual_height_range": [float(np.min(actual[:, 0])), float(np.max(actual[:, 0]))],
        "actual_distance_range": [float(np.min(actual[:, 1])), float(np.max(actual[:, 1]))],
        "actual_output_convex_hull": convex_hull(actual),
        "affine_fit": affine_metrics,
        "local_jacobian": {
            "min_abs_determinant": float(np.min(np.abs(determinants))),
            "median_abs_determinant": float(np.median(np.abs(determinants))),
            "max_condition_number": float(np.max(conditions)),
            "nonpositive_determinant_cells": int(np.sum(determinants <= 0.0)),
            "cell_count": len(jacobians),
        },
    }

    if args.heldout:
        inverse_lut = np.asarray([
            [row["actual_height"], row["actual_distance"], row["command_height"], row["command_distance"]]
            for row in lut_rows
        ])
        validation, metrics = validate_heldout(surface, inverse_lut, load_rows(args.heldout))
        write_csv(os.path.join(args.output_dir, "heldout_validation.csv"), validation)
        report["heldout_validation"] = metrics

    report_path = os.path.join(args.output_dir, "calibration_report.json")
    with open(report_path, "w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
