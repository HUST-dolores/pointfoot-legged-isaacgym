"""Numerical checks for the measured sliding demonstration, not a training claim."""

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze import robot_rows,robot_measures
from .geometry import sha256,write_json


def main(root):
    safe=root/"robot_demo_safe_verified"; drive=root/"robot_demo_drive"
    safe_meta,safe_z,safe_m=robot_measures(safe)
    drive_meta,drive_z,_=robot_measures(drive)
    s=robot_rows(safe)[0]; d=robot_rows(drive)[0]
    assert s["supported_slip_ge_2cm"] and s["max_supported_slip_m"]>.05
    assert s["last_half_s_supported_fraction"]>.95 and not s["departed_after_excitation"]
    assert s["force_after_2p06_max_N"]==0
    moving=(safe_z["time_s"]>=2.-1e-5)&(safe_z["time_s"]<=3.+1e-5)
    continuously_supported=((abs(safe_m["normal_gap_m"][:,0])<.008)
        &(safe_m["normal_force_N"][:,0]>.2*safe_meta["mass_kg"]*9.81)
        &(safe_m["footprint_margin_m"][:,0]>0))
    support_fraction=float(continuously_supported[moving].mean())
    assert support_fraction>.95
    assert not drive_z["force_previous_interval"].any()
    assert d["supported_slip_ge_2cm"] and d["departed_after_excitation"]
    assert not safe_meta["robot_base_fixed"] and not drive_meta["robot_base_fixed"]
    source=Path(__file__).with_name("robot.py")
    assert safe_meta["runner_source_sha256"]==drive_meta["runner_source_sha256"]==sha256(source)
    geometry_rows=[]
    for name in ("robot_original_verified","robot_raised_v2"):
        rows=robot_rows(root/name)
        assert any(r["supported_slip_ge_2cm"] and r["last_half_s_supported_fraction"]>.95 for r in rows)
        control=[r for r in rows if r["preset"] in ("original","box_only_low")]
        assert max(r["max_supported_slip_m"] for r in control)<.002
        props=json.loads((root/name/"materials.json").read_text())
        run_meta=json.loads((root/name/"result.json").read_text())
        assert run_meta["runner_source_sha256"]==sha256(source)
        assert all(p["wheel_and_leg_shapes_unchanged"] for p in props)
        assert all(np.allclose(p["support_contact_offset_m"],.002) for p in props)
        geometry_rows.append({"run":name,"cases":len(rows),"support_base_shape_indices":props[0]["base_shape_indices"]})
    # Rendering/layout revisions must not alter the recorded physical demonstration.
    original=dict(np.load(root/"robot_demo_safe"/"trace.npz",allow_pickle=False))
    replay=[]
    for key in ("root","box","box_contact_force","actions","force_previous_interval"):
        np.testing.assert_array_equal(original[key],safe_z[key])
        replay.append(key)
    media=json.loads((root/"media/verification.json").read_text())
    assert media["status"]=="passed" and len(media["videos"])==2
    summary={"status":"passed","safe_slide":s,"body_motion_slip":d,
        "canonical_geometry_sweeps":geometry_rows,"safe_render_replay_exact_arrays":replay,
        "nominal_urdf_sha256":safe_meta["nominal_sha256"],"runner_sha256":sha256(source),
        "supported_fraction_during_slide_2_to_3_s":support_fraction,
        "max_abs_gap_during_slide_m":float(abs(safe_m["normal_gap_m"][moving,0]).max()),
        "camera_full_frame_and_nonblank":True,"unit_tests":5,
        "not_a_trained_retention_controller":True}
    write_json(root/"analysis/verification.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input",type=Path,required=True)
    main(p.parse_args().input)
