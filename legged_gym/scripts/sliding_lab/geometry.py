"""Plate-relative measurements, including box footprint rather than only its center."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


CORNERS = np.array([[x,y,z] for x in (-.05,.05) for y in (-.05,.05) for z in (-.05,.05)],dtype=np.float64)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""): h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def urdf_audit(path):
    robot=ET.parse(path).getroot()
    base=robot.find("link[@name='base_Link']/collision")
    origin=np.fromstring(base.find("origin").get("xyz"),sep=" ")
    size=np.fromstring(base.find("geometry/box").get("size"),sep=" ")
    joint=robot.find("joint[@name='plate_fixed_joint']/origin")
    center=np.fromstring(joint.get("xyz"),sep=" ")
    plate=np.fromstring(robot.find("link[@name='plate_link']/collision/geometry/box").get("size"),sep=" ")
    return {"base_collision_top_z_m":float(origin[2]+size[2]/2),
        "plate_center_xyz_m":center.tolist(),"plate_size_m":plate.tolist(),
        "plate_top_z_m":float(center[2]+plate[2]/2),
        "plate_above_base_top_m":float(center[2]+plate[2]/2-origin[2]-size[2]/2)}


def make_variant(original,dest,plate_z):
    tree=ET.parse(original)
    tree.getroot().find("joint[@name='plate_fixed_joint']/origin").set("xyz",f"0 0 {plate_z:.6f}")
    # Keep the nominal file untouched; generated sibling retains relative mesh paths.
    tree.write(dest,encoding="utf-8",xml_declaration=True)


def measure(box_position,box_rotation,plate_position,plate_rotation,force_world):
    delta=box_position-plate_position
    relative=np.einsum("...ji,...j->...i",plate_rotation,delta)
    orientation=np.einsum("...ji,...jk->...ik",plate_rotation,box_rotation)
    corners=relative[...,None,:]+np.einsum("...ij,cj->...ci",orientation,CORNERS)
    margin=np.minimum(.25-np.abs(corners[...,0]).max(-1),.25-np.abs(corners[...,1]).max(-1))
    gap=corners[...,2].min(-1)-.0025
    normal_force=np.einsum("...i,...i->...",force_world,plate_rotation[...,2])
    tilt=np.rad2deg(np.arccos(np.clip(orientation[...,2,2],-1,1)))
    bottom=relative+np.einsum("...ij,j->...i",orientation,np.array([0.,0.,-.05]))
    return {"center_plate":relative,"bottom_center_plate":bottom,"footprint_margin_m":margin,
            "normal_gap_m":gap,"normal_force_N":normal_force,"relative_tilt_deg":tilt}
