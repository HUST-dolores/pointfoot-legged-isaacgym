"""Measure supported slip separately from tipping, airborne motion and departure."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import measure,write_json


def csv_write(path,rows):
    if not rows: return
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def robot_measures(directory):
    meta=json.loads((directory/"result.json").read_text())
    z=dict(np.load(directory/"trace.npz",allow_pickle=False))
    root=z["root"]
    R=Rotation.from_quat(root[...,3:7].reshape(-1,4)).as_matrix().reshape(*root.shape[:2],3,3)
    p=root[...,:3]+np.einsum("...ij,j->...i",R,np.array(meta["geometry"]["plate_center_xyz_m"]))
    boxR=Rotation.from_quat(z["box"][...,3:7].reshape(-1,4)).as_matrix().reshape(*root.shape[:2],3,3)
    m=measure(z["box"][...,:3],boxR,p,R,z["box_contact_force"])
    return meta,z,m


def robot_rows(directory):
    meta,z,m=robot_measures(directory)
    rows=[]; t=z["time_s"]; reference=np.argmin(abs(t-1.98))
    for case in meta["cases"]:
        i=case["case_id"]
        d=np.linalg.norm(m["bottom_center_plate"][:,i,:2]-m["bottom_center_plate"][reference,i,:2],axis=-1)
        supported=(abs(m["normal_gap_m"][:,i])<.008)&(m["footprint_margin_m"][:,i]>0)
        supported&=(m["normal_force_N"][:,i]>.2*meta["mass_kg"]*9.81)&(m["relative_tilt_deg"][:,i]<12)
        supported&=z["has_load"][:,i]&~z["first_failure"][:,i]
        post=t>=2.-1e-5; candidate=supported&post
        last=(t>=t[-1]-.5)&supported
        slipping=candidate&(d>=.02)
        times=np.flatnonzero(slipping)
        whole_out=((abs(m["center_plate"][:,i,0])>.25+np.sqrt(3)*.05)
                   |(abs(m["center_plate"][:,i,1])>.25+np.sqrt(3)*.05)
                   |(m["normal_gap_m"][:,i]<-.1))&post
        edge=np.flatnonzero(post&(m["footprint_margin_m"][:,i]<-.001))
        gone=np.flatnonzero(whole_out)
        rows.append({"run":directory.name,"geometry":case["geometry"],"preset":case["preset"],
            "direction":case["direction"],"excitation":case["excitation"],"case_id":i,
            "max_supported_slip_m":float(d[candidate].max()) if candidate.any() else 0.,
            "supported_slip_ge_2cm":bool(times.size),
            "first_supported_2cm_s":float(t[times[0]]) if times.size else None,
            "last_half_s_supported_fraction":float(last.sum()/max(1,(t>=t[-1]-.5).sum())),
            "departed_after_excitation":bool(whole_out.any()),
            "first_edge_crossing_s":float(t[edge[0]]) if edge.size else None,
            "first_fully_departed_s":float(t[gone[0]]) if gone.size else None,
            "first_termination":bool(z["first_failure"][:,i].any()),
            "median_normal_force_during_slip_N":float(np.median(m["normal_force_N"][slipping,i])) if slipping.any() else None,
            "median_gap_during_slip_m":float(np.median(m["normal_gap_m"][slipping,i])) if slipping.any() else None,
            "force_max_N":float(np.linalg.norm(z["force_previous_interval"][:,i],axis=-1).max()),
            "force_after_2p06_max_N":float(np.linalg.norm(z["force_previous_interval"][t>2.06,i],axis=-1).max())})
    return rows


def bench_rows(directory):
    z=dict(np.load(directory/"trace.npz",allow_pickle=False)); rows=json.loads((directory/"cases.json").read_text())
    t=z["time_s"]; out=[]
    for case in rows:
        i=case["case_id"]
        p=z["plate_root"][:,i]
        R=Rotation.from_quat(p[:,3:7]).as_matrix()
        B=Rotation.from_quat(z["root"][:,i,3:7]).as_matrix()
        m=measure(z["root"][:,i,:3],B,p[:,:3],R,z["contact_force"][:,i])
        ref=np.argmin(abs(t-(.4 if case["mechanism"]=="incline" else .995)))
        d=np.linalg.norm(m["bottom_center_plate"][:,:2]-m["bottom_center_plate"][ref,:2],axis=-1)
        supported=(abs(m["normal_gap_m"])<.008)&(m["footprint_margin_m"]>0)&(m["relative_tilt_deg"]<12)
        supported&=m["normal_force_N"]>.2*case["mass_kg"]*9.81
        supported&=t>=(.4 if case["mechanism"]=="incline" else 1.)
        out.append({"case_id":i,"mechanism":case["mechanism"],"plate_friction":case["plate_friction"],
            "box_friction":case["box_friction"],"repeat":case["repeat"],
            "max_supported_slip_m":float(d[supported].max()) if supported.any() else 0.,
            "final_x_plate_m":float(m["center_plate"][-1,0]),"final_gap_m":float(m["normal_gap_m"][-1])})
    return out


def main(root):
    out=root/"analysis"; out.mkdir(exist_ok=True)
    rows=[]
    for directory in sorted(root.glob("robot*")):
        if (directory/"result.json").exists(): rows.extend(robot_rows(directory))
    csv_write(out/"robot_slip_cases.csv",rows)
    bench=root/"bench_v2"
    if bench.exists(): csv_write(out/"bench_slip_cases.csv",bench_rows(bench))
    write_json(out/"summary.json",{"robot_cases":rows,"supported_slip_definition":
        "bottom-center moves >=2 cm in plate frame, full footprint inside, gap <8mm, normal load >0.2mg, relative tilt<12deg, before original termination"})
    for r in rows:
        print(r["run"],r["preset"],r["direction"],f"slip={r['max_supported_slip_m']:.4f}",
              f"supported_end={r['last_half_s_supported_fraction']:.2f}","departed="+str(r["departed_after_excitation"]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input",type=Path,required=True)
    main(p.parse_args().input)
