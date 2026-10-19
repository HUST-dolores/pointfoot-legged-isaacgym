"""Encode unmodified simulation frames and plot actual plate-frame measurements."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

from .analyze import robot_measures,csv_write
from .geometry import write_json


def main(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    media=root/"media"; media.mkdir(exist_ok=True)
    outputs=[]
    for name,label in (("robot_demo_safe_verified","safe_supported_slide"),("robot_demo_drive","body_motion_slip")):
        folder=root/name
        if not (folder/"result.json").exists(): continue
        meta,z,m=robot_measures(folder); frames=sorted((folder/"frames").glob("*.png"))
        if not frames: raise ValueError("missing actual simulator frames")
        bounds=[]; changed=[]; previous=None
        for path in frames:
            with Image.open(path) as image:
                rgb=np.asarray(image.convert("RGB"))
            if rgb.std()<3: raise ValueError("blank simulation image")
            mask=rgb.max(-1).astype(int)-rgb.min(-1).astype(int)>8
            yy,xx=np.where(mask)
            if not len(xx) or xx.min()<5 or xx.max()>=rgb.shape[1]-5 or yy.min()<5 or yy.max()>=rgb.shape[0]-5:
                raise ValueError(f"colored geometry touches the frame border: {path}")
            bounds.append([int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())])
            if previous is not None: changed.append(float((abs(rgb.astype(int)-previous)>10).any(-1).mean()))
            previous=rgb.astype(int)
        if max(changed)<1e-4: raise ValueError("motionless image sequence")
        target=media/f"{label}.mp4"
        subprocess.run(["ffmpeg","-y","-v","error","-framerate","10","-pattern_type","glob","-i",str(folder/"frames/*.png"),
            "-c:v","libx264","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",str(target)],check=True,timeout=120)
        for time_s,tag in ((1.92,"before"),(2.72,"after"),(5.92,"end")):
            source=min(frames,key=lambda p:abs(float(p.stem[2:])-time_s))
            shutil.copy2(source,media/f"{label}_{tag}.png")
        t=z["time_s"]; valid=z["has_load"][:,0]
        bodyxy=m["bottom_center_plate"][:,0,:2]
        rows=[]
        for i,time_s in enumerate(t):
            if not valid[i]: continue
            velocity=(bodyxy[i]-bodyxy[i-1])/(float(t[i])-float(t[i-1])) if i and valid[i-1] else None
            rows.append({"time_s":float(time_s),"bottom_x_plate_m":float(bodyxy[i,0]),"bottom_y_plate_m":float(bodyxy[i,1]),
                "vx_rel_fd_mps":float(velocity[0]) if velocity is not None else None,
                "vy_rel_fd_mps":float(velocity[1]) if velocity is not None else None,
                "normal_gap_m":float(m["normal_gap_m"][i,0]),"normal_force_N":float(m["normal_force_N"][i,0]),
                "footprint_margin_m":float(m["footprint_margin_m"][i,0]),"relative_tilt_deg":float(m["relative_tilt_deg"][i,0]),
                "box_force_N":float(np.linalg.norm(z["force_previous_interval"][i,0])),
                "root_x_m":float(z["root"][i,0,0]),"root_y_m":float(z["root"][i,0,1])})
        csv_write(media/f"{label}_measurements.csv",rows)
        plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False})
        fig,axes=plt.subplots(4,1,figsize=(10,8),sharex=True)
        axes[0].plot(t,np.where(valid,bodyxy[:,0]*100,np.nan),label="x",color="#337F9F")
        axes[0].plot(t,np.where(valid,bodyxy[:,1]*100,np.nan),label="y",color="#BC703C")
        axes[0].set_ylabel("Bottom center (cm)"); axes[0].legend(frameon=False,ncol=2,loc="upper left")
        axes[1].plot(t,np.where(valid,m["normal_gap_m"][:,0]*1000,np.nan),color="#6F66A0")
        axes[1].set_ylabel("Normal gap (mm)")
        axes[2].plot(t,np.where(valid,m["normal_force_N"][:,0]/(meta["mass_kg"]*9.81),np.nan),color="#2E8C76")
        axes[2].set_ylabel("Normal force / mg")
        axes[3].plot(t,np.where(valid,m["footprint_margin_m"][:,0]*100,np.nan),color="#A34C4C")
        axes[3].axhline(0,color="#717B84",ls="--",lw=.8); axes[3].set_ylabel("Edge margin (cm)")
        for ax in axes:
            ax.axvline(2.,color="#9099A0",ls=":",lw=.8); ax.grid(axis="y",color="#E0E5E8",lw=.6)
        axes[-1].set_xlabel("Time (s)"); axes[-1].set_xlim(0,6)
        fig.subplots_adjust(left=.17,right=.98,bottom=.08,top=.97,hspace=.22)
        fig.savefig(media/f"{label}_measurements.png",dpi=180); fig.savefig(media/f"{label}_measurements.pdf")
        plt.close(fig)
        outputs.append({"directory":name,"video":target.name,"frames":len(frames),
            "colored_geometry_bounds_all_frames":[np.min(bounds,axis=0)[:2].tolist(),np.max(bounds,axis=0)[2:].tolist()],
            "max_adjacent_changed_pixel_fraction":max(changed),"original_frames_not_edited":True,
            "all_box_forces_zero":bool(not z["force_previous_interval"].any())})
    write_json(media/"verification.json",{"status":"passed","videos":outputs})
    print(json.dumps(outputs,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input",type=Path,required=True)
    main(p.parse_args().input)
