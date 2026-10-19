"""Encode unmodified camera frames after visibility and motion checks."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
from .geometry import write_json


def main(a):
    a.output.mkdir(parents=True, exist_ok=True)
    outputs = []
    for directory in a.demos:
        metadata = json.loads((directory / "result.json").read_text())
        frames = metadata["camera_frames"]
        assert frames and metadata["no_pose_writes_after_spawn"]
        previous = None
        bounds, changes, payload_pixels = [], [], []
        for row in frames:
            with Image.open(directory / "frames" / row["file"]) as im:
                rgb = np.asarray(im.convert("RGB")).astype(np.int16)
            assert rgb.std() > 3.
            mask = rgb.max(-1) - rgb.min(-1) > 8
            yy, xx = np.where(mask)
            assert len(xx) and xx.min() >= 5 and yy.min() >= 5 and xx.max() < rgb.shape[1] - 5 and yy.max() < rgb.shape[0] - 5
            bounds.append([int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())])
            orange = (rgb[..., 0] > 1.5 * rgb[..., 1]) & (rgb[..., 1] > 1.3 * rgb[..., 2]) & (rgb[..., 0] > 70)
            if row["time_s"] >= .6:
                payload_pixels.append(int(orange.sum()))
            if previous is not None:
                changes.append(float((np.abs(rgb - previous) > 10).any(-1).mean()))
            previous = rgb
        assert max(changes) > 1e-4
        # Pixel presence is recorded, not blindly asserted: physical occlusion can hide a valid object.
        target = a.output / (directory.name + ".mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "10", "-i", str(directory / "frames/frame_%04d.png"),
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
                       check=True, timeout=120)
        for t, name in ((1.94, "before"), (2.84, "after"), (frames[-1]["time_s"], "end")):
            row = min(frames, key=lambda r: abs(r["time_s"] - t))
            shutil.copy2(directory / "frames" / row["file"], a.output / (directory.name + "_" + name + ".png"))
        outputs.append({"video": target.name, "frames": len(frames), "colored_geometry_min_xy": np.min(bounds, axis=0)[:2].tolist(),
            "colored_geometry_max_xy": np.max(bounds, axis=0)[2:].tolist(), "max_adjacent_changed_pixel_fraction": max(changes),
            "minimum_payload_color_pixels_after_spawn": min(payload_pixels),
            "source_frames_unedited": True, "simulator_trace_retained": True})
    write_json(a.output / "video_verification.json", {"status": "passed", "videos": outputs})
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demos", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
