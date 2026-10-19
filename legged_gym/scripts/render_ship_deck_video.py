"""Render CPU PhysX deck motion with native cameras or a software fallback.

Three physical actors: roll-only, pitch-only, and roll+pitch+heave. Uses the
same motion_state() implementation as wheelfoot_ship. Software rendering is
explicitly labelled; no robot or balancing success is implied by the video.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from isaacgym import gymapi, gymtorch
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
# Load the same math file without triggering the project's utils/env circular import.
spec = importlib.util.spec_from_file_location(
    "ship_video_motion", Path(__file__).resolve().parents[1]/"utils/ship_motion.py")
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)
motion_state, rotate, surface_pose = motion.motion_state, motion.rotate, motion.surface_pose


def ffmpeg_environment():
    """Do not let simulator/Conda libav override the system encoder libraries."""
    environment=os.environ.copy()
    environment.pop('LD_LIBRARY_PATH',None)
    return environment


def euler(root):
    x, y, z, w = root[..., 3:7].unbind(-1)
    roll = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = torch.asin((2*(w*y-z*x)).clamp(-1, 1))
    return torch.stack((roll, pitch), dim=-1)


def simulate(args):
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 0.005
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.use_gpu_pipeline = False
    params.physx.use_gpu = False
    params.physx.num_threads = 2
    sim = gym.create_sim(0, 0 if args.native else -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("CPU PhysX simulation could not be created")
    try:
        options = gymapi.AssetOptions()
        options.disable_gravity = True
        options.fix_base_link = False
        options.density = 10000/(6*4*0.2)
        options.linear_damping = 0
        options.angular_damping = 0
        asset = gym.create_box(sim, 6, 4, 0.2, options)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.4)
        ids, cameras, origins = [], [], []
        if args.native:
            plane = gymapi.PlaneParams()
            plane.normal = gymapi.Vec3(0,0,1)
            # PhysX plane equation n.x + distance = 0: +1 means z=-1.
            plane.distance = 1.0  # below the entire deck-motion envelope
            gym.add_ground(sim,plane)
        for i in range(3):
            env = gym.create_env(sim, gymapi.Vec3(-12,-12,0), gymapi.Vec3(12,12,4), 3)
            actor = gym.create_actor(env, asset, pose, f"deck_{i}", i, 0, 0)
            ids.append(gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM))
            o = gym.get_env_origin(env)
            origins.append([o.x,o.y,o.z])
            gym.set_rigid_body_color(env,actor,0,gymapi.MESH_VISUAL_AND_COLLISION,gymapi.Vec3(.15,.58,.64))
            if args.native:
                properties = gymapi.CameraProperties()
                properties.width, properties.height = 400,260
                properties.horizontal_fov = 50
                camera = gym.create_camera_sensor(env,properties)
                if camera < 0:
                    raise RuntimeError("native camera creation failed")
                # Camera locations are world coordinates, unlike actor spawn poses.
                gym.set_camera_location(camera,env,gymapi.Vec3(o.x+6,o.y-8,o.z+5),
                                        gymapi.Vec3(o.x,o.y,o.z+1.4))
                cameras.append((env,camera))
        gym.prepare_sim(sim)
        root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        indices = torch.tensor(ids, dtype=torch.int32)
        amp = torch.tensor([[math.radians(args.roll_deg),0,0],
                            [0,math.radians(args.pitch_deg),0],
                            [math.radians(args.roll_deg),math.radians(args.pitch_deg),args.heave_m]])
        freq = torch.tensor([[0.25,0.18,0.32]]).repeat(3,1)
        phase = torch.tensor([[0.,1.1,0.8]]).repeat(3,1)
        origins = torch.tensor(origins)
        pivot = origins + torch.tensor([[0.,0.,1.5]])
        # Initialize all actors in the same world-coordinate convention used by
        # subsequent tensor updates before recording the first frame.
        initial, _, _ = motion_state(torch.zeros(3),amp,freq,phase,pivot,0.2,2.0)
        root[:] = initial
        gym.set_actor_root_state_tensor_indexed(sim,gymtorch.unwrap_tensor(root),
                                               gymtorch.unwrap_tensor(indices),3)
        gym.simulate(sim)
        gym.fetch_results(sim,True)
        n = int(args.duration*args.fps)
        samples, times, camera_frames = [], [], []
        next_frame = 0
        max_position_error = 0.0
        for step in range(int(math.ceil(args.duration/params.dt))+1):
            t = step*params.dt
            gym.refresh_actor_root_state_tensor(sim)
            target, _, _ = motion_state(torch.full((3,), t), amp, freq, phase, pivot, 0.2, 2.0)
            max_position_error = max(max_position_error, float((root[:, :3]-target[:, :3]).norm(dim=-1).max()))
            if t + 1e-8 >= next_frame/args.fps:
                if args.native:
                    assert all(len(gym.get_env_rigid_contacts(env)) == 0 for env,_ in cameras), \
                        "reference plane unexpectedly contacts a deck"
                samples.append(root.clone())
                times.append(t)
                if args.native:
                    gym.step_graphics(sim)
                    gym.render_all_camera_sensors(sim)
                    camera_frames.append(np.stack([
                        gym.get_camera_image(sim,env,camera,gymapi.IMAGE_COLOR).reshape(260,400,4)[:, :, :3].copy()
                        for env,camera in cameras]))
                next_frame += 1
                if next_frame == n:
                    break
            root[:] = target
            gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root),
                                                   gymtorch.unwrap_tensor(indices), 3)
            gym.simulate(sim)
            gym.fetch_results(sim, True)
        poses = torch.stack(samples)
        if max_position_error > 0.001:
            raise RuntimeError(f"unexpected deck tracking error: {max_position_error} m")
        poses[..., :3] -= origins[None]
        angles = euler(poses)*180/math.pi
        top, _ = surface_pose(poses, 0.2)
        report = {
            "source": ("CPU PhysX + Isaac Gym native cameras" if args.native else "actual CPU PhysX actor root state; software-rendered replay"),
            "root_states_frame": "environment-local; environment translation offset subtracted",
            "reference_plane_z_m": -1.0 if args.native else None,
            "same_motion_module_as_ship_task": True,
            "robot_present": False,
            "deck_size_m": [6,4,0.2], "deck_mass_kg": 10000,
            "gravity_m_s2": [0,0,-9.81], "physics_dt_s": params.dt,
            "frames": len(samples), "fps": args.fps,
            "roll_max_abs_deg": angles[..., 0].abs().max(dim=0).values.tolist(),
            "pitch_max_abs_deg": angles[..., 1].abs().max(dim=0).values.tolist(),
            "heave_max_abs_m": (top[..., 2]-1.5).abs().max(dim=0).values.tolist(),
            "max_position_tracking_error_m": max_position_error,
            "demonstration_amplitudes": True,
            "balance_success_claim": False,
        }
        return np.asarray(times), poses.numpy(), angles.numpy(), (top[...,2]-1.5).numpy(), report, camera_frames
    finally:
        gym.destroy_sim(sim)


FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"


def render(args, times, poses, angles, heave, camera_frames):
    fonts = {size: ImageFont.truetype(FONT, size, index=2) for size in (16,18,20,22,26,32)}
    vertices = torch.tensor([[-3,-2,-.1],[3,-2,-.1],[3,2,-.1],[-3,2,-.1],
                             [-3,-2,.1],[3,-2,.1],[3,2,.1],[-3,2,.1]], dtype=torch.float32)
    q = torch.from_numpy(poses[..., 3:7])
    world = rotate(q[:, :, None, :].expand(-1,-1,8,-1),
                   vertices[None,None].expand(len(times),3,-1,-1))
    world += torch.from_numpy(poses[..., :3])[:, :, None]
    world = world.numpy()
    look = np.asarray([-8.,10.,-6.])
    look /= np.linalg.norm(look)
    right = np.cross(look, [0.,0.,1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, look)
    origin = np.asarray([0.,0.,1.5])
    faces = [(0,1,2,3),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7),(4,5,6,7)]
    colors = [(40,72,82),(60,98,110),(75,115,125),(60,98,110),(75,115,125),(40,151,162)]
    labels = ["① 横摇：绕 X 轴", "② 纵摇：绕 Y 轴", "③ 横摇 + 纵摇 + 升沉"]
    curve_colors = [(226,98,69),(72,128,221),(56,160,120)]
    output = args.output/"ship_deck_roll_pitch_heave.mp4"
    encoder = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1280x720", "-r", str(args.fps), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output)], stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        env=ffmpeg_environment())
    try:
        for frame in range(len(times)):
            im = Image.new("RGB", (1280,720), (239,244,248))
            draw = ImageDraw.Draw(im)
            draw.text((26,15), "甲板运动：横摇、纵摇与升沉", font=fonts[32], fill=(28,44,62))
            subtitle = "CPU PhysX + Isaac Gym 原生相机 · 无机器人" if args.native else "CPU PhysX 实测位姿回放 · 软件渲染 · 无机器人"
            draw.text((28,64), subtitle, font=fonts[18], fill=(89,104,120))
            draw.text((1080,27), f"t = {times[frame]:5.2f} s", font=fonts[22], fill=(28,44,62))
            for panel in range(3):
                left = 20+420*panel
                draw.rounded_rectangle((left,106,left+400,645), radius=16, fill=(255,255,255))
                draw.text((left+16,122), labels[panel], font=fonts[22], fill=(28,44,62))
                draw.text((left+16,163), f"横摇 {angles[frame,panel,0]:+6.1f}°   纵摇 {angles[frame,panel,1]:+6.1f}°",
                          font=fonts[18], fill=(67,83,100))
                draw.text((left+16,193), f"升沉 {100*heave[frame,panel]:+6.1f} cm", font=fonts[18], fill=(67,83,100))

                def project(points):
                    points = np.asarray(points)-origin
                    return np.column_stack((left+200+43*(points@right), 358-43*(points@up)))

                # A fixed visual coordinate grid, not a collider in the simulation.
                for x in np.arange(-4,4.1,1):
                    p = project([[x,-3,0],[x,3,0]])
                    draw.line([tuple(v) for v in p], fill=(223,232,240), width=1)
                for y in np.arange(-3,3.1,1):
                    p = project([[-4,y,0],[4,y,0]])
                    draw.line([tuple(v) for v in p], fill=(223,232,240), width=1)
                ref = project([[-3,-2,1.5],[3,-2,1.5],[3,2,1.5],[-3,2,1.5],[-3,-2,1.5]])
                draw.line([tuple(v) for v in ref], fill=(224,172,102), width=2)
                pts = world[frame,panel]
                screen = project(pts)
                for face_id in sorted(range(6), key=lambda j: float(pts[list(faces[j])].mean(0)@look), reverse=True):
                    polygon = [tuple(screen[v]) for v in faces[face_id]]
                    draw.polygon(polygon, fill=colors[face_id], outline=(34,77,88))
                # Mark the longitudinal direction on the actual moving top face.
                local = torch.tensor([[-1.6,0,.105],[1.6,0,.105],[1.1,.25,.105],[1.6,0,.105],[1.1,-.25,.105]])
                points = rotate(q[frame,panel].expand(5,-1), local).numpy()+poses[frame,panel,:3]
                arrow = project(points)
                draw.line([tuple(arrow[0]),tuple(arrow[1])], fill=(239,252,255), width=4)
                draw.line([tuple(v) for v in arrow[2:]], fill=(239,252,255), width=4)
                surface = project([poses[frame,panel,:3]+rotate(q[frame,panel],torch.tensor([0.,0.,.1])).numpy()])[0]
                draw.ellipse((surface[0]-5,surface[1]-5,surface[0]+5,surface[1]+5), fill=(249,191,70))
                # Fixed world-axis triad makes the direction of rocking unambiguous.
                axes = np.asarray([[-3.4,-2.4,0],[-1.9,-2.4,0],[-3.4,-.9,0],[-3.4,-2.4,1.5]])
                triad = project(axes)
                for k,color in enumerate(curve_colors, start=1):
                    draw.line([tuple(triad[0]),tuple(triad[k])], fill=color, width=3)
                    draw.text(tuple(triad[k]), ["X","Y","Z"][k-1],font=fonts[16],fill=color)
                if args.native:
                    draw.rectangle((left+1,224,left+399,525),fill=(255,255,255))
                    im.paste(Image.fromarray(camera_frames[frame][panel]),(left,224))
                channel = angles[:,panel,0] if panel == 0 else angles[:,panel,1] if panel == 1 else heave[:,panel]*100
                limit = args.roll_deg if panel == 0 else args.pitch_deg if panel == 1 else args.heave_m*100
                label = "横摇 / deg" if panel == 0 else "纵摇 / deg" if panel == 1 else "综合运动的升沉 / cm"
                draw.text((left+16,492), label,font=fonts[18],fill=curve_colors[panel])
                px = left+18+(times/args.duration)*364
                py = 573-(channel/max(limit,1e-6))*37
                draw.line((left+18,573,left+382,573), fill=(221,229,237),width=1)
                draw.line([tuple(v) for v in np.column_stack((px,py))], fill=(208,218,228),width=2)
                draw.line([tuple(v) for v in np.column_stack((px[:frame+1],py[:frame+1]))], fill=curve_colors[panel],width=3)
                draw.line((px[frame],526,px[frame],621),fill=(97,111,126),width=1)
                draw.ellipse((px[frame]-4,py[frame]-4,px[frame]+4,py[frame]+4),fill=curve_colors[panel])
            draw.text((25,661), f"演示幅值：横摇 ±{args.roll_deg:g}° / 纵摇 ±{args.pitch_deg:g}° / 升沉 ±{100*args.heave_m:g} cm",
                      font=fonts[20],fill=(28,44,62))
            footer = ("棋盘地面为固定参考；仅展示甲板运动，不表示机器人已能稳定平衡。" if args.native else
                      "橙框与灰网格为固定参考；仅展示甲板运动，不表示机器人已能稳定平衡。")
            draw.text((25,693), footer,
                      font=fonts[16],fill=(89,104,120))
            if frame == min(len(times)-1,3*args.fps):
                im.save(args.output/"preview.png")
            if frame == 0:
                im.save(args.output/"preview_start.png")
            encoder.stdin.write(im.tobytes())
            if frame % 100 == 0:
                print(f"Rendered {frame}/{len(times)} frames", flush=True)
        encoder.stdin.close()
        error = encoder.stderr.read().decode()
        code = encoder.wait()
        if code:
            raise RuntimeError(f"ffmpeg exited {code}: {error}")
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", action="store_true", help="use Isaac Gym native GPU cameras with CPU physics")
    parser.add_argument("--duration", type=float, default=16)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--roll-deg", type=float, default=20)
    parser.add_argument("--pitch-deg", type=float, default=12)
    parser.add_argument("--heave-m", type=float, default=0.15)
    args = parser.parse_args()
    if args.duration < 4 or not 1 <= args.fps <= 200 or not 0 < args.roll_deg < 60 or not 0 < args.pitch_deg < 60 or args.heave_m <= 0:
        parser.error("duration >=4, fps 1..200, angles 0..60 and positive heave required")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    times, poses, angles, heave, report, camera_frames = simulate(args)
    np.savez_compressed(args.output/"actual_deck_poses.npz",time_s=times,root_states=poses,
                        roll_pitch_deg=angles,heave_m=heave)
    output = render(args,times,poses,angles,heave,camera_frames)
    report["video"] = str(output)
    (args.output/"video_metadata.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
