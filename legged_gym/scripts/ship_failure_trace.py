"""Run ship_motion_probe unchanged, recording physical states BEFORE fall reset.

Accepts the same CLI as ship_motion_probe. Capture is read-only, does not add
policy observations, and excludes startup/regression events before sea time 2s.
A local replay is diagnostic evidence, not a replacement for matched lab tests.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from legged_gym.scripts import ship_motion_probe as probe
from legged_gym.utils.ship_motion import surface_pose, inverse_rotate

TRACES = []

class FailureTraceShip(probe.BipedWFShip):
    def reset_idx(self, env_ids):
        if len(env_ids) and hasattr(self, 'deck_indices'):
            capture = (self.reset_buf[env_ids].bool()
                       & ~self.time_out_buf[env_ids].bool()
                       & (self.ship_time[env_ids] > 2.0))
            chosen = env_ids[capture]
            if chosen.numel():
                robot = self.root_states[self.actor_indices[chosen]].detach().clone()
                deck = self.root_states[self.deck_indices[chosen]].detach().clone()
                top, normal = surface_pose(deck, self.cfg.ship.thickness)
                height = ((robot[:, :3]-top)*normal).sum(-1)
                gravity = inverse_rotate(robot[:, 3:7], self.gravity_vec[chosen])
                physical_dof = self.dof_state.view(self.num_envs, self.num_dof, 2)[chosen].detach().clone()
                feet = self.foot_positions[chosen].detach().clone()
                feet_contact = self.contact_forces[chosen][:, self.feet_indices].norm(dim=-1)
                half = [self.cfg.ship.length/2, self.cfg.ship.width/2]
                for row, env_id in enumerate(chosen.cpu().tolist()):
                    cached_height = float(self.base_height[env_id])
                    feet_deck = inverse_rotate(deck[row:row+1, 3:7].expand(feet.shape[1], -1), feet[row]-top[row:row+1])
                    TRACES.append({
                        'env_id': env_id,
                        'episode_time_s': float(self.episode_length_buf[env_id])*self.dt,
                        'sea_time_s': float(self.ship_time[env_id]),
                        'physical_height_m': float(height[row]),
                        'cached_height_m': cached_height,
                        'height_cache_error_m': abs(float(height[row])-cached_height),
                        'gravity_cache_error': float((gravity[row]-self.projected_gravity[env_id]).norm()),
                        'causes': {k: bool(v[env_id]) for k, v in self.ship_termination_causes.items()},
                        'robot_root': robot[row].cpu().tolist(),
                        'deck_root': deck[row].cpu().tolist(),
                        'physical_dof_position': physical_dof[row, :, 0].cpu().tolist(),
                        'physical_dof_velocity': physical_dof[row, :, 1].cpu().tolist(),
                        'actions': self.actions[env_id].detach().cpu().tolist(),
                        'feet_deck_xyz_m': feet_deck.cpu().tolist(),
                        'foot_contact_n': feet_contact[row].cpu().tolist(),
                        'deck_half_extents_m': half,
                        'any_foot_center_outside_xy': bool(((feet_deck[:,0].abs()>half[0]) | (feet_deck[:,1].abs()>half[1])).any()),
                    })
        return super().reset_idx(env_ids)

def main():
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument('--output', type=Path)
    args, _ = output_parser.parse_known_args()
    if args.output is None:
        probe.main()
        return
    original = probe.BipedWFShip
    probe.BipedWFShip = FailureTraceShip
    TRACES.clear()
    try:
        probe.main()
    finally:
        probe.BipedWFShip = original
        if args.output.exists():
            report = {
                'scope': 'read-only terminal-state trace; not balance success evidence',
                'sampling': 'physical post-step states before inherited reset',
                'minimum_sea_time_s': 2.0,
                'recorded_failures': len(TRACES),
                'max_height_cache_error_m': max((x['height_cache_error_m'] for x in TRACES), default=None),
                'max_gravity_cache_error': max((x['gravity_cache_error'] for x in TRACES), default=None),
                'events': TRACES,
            }
            (args.output/'failure_trace.json').write_text(json.dumps(report, indent=2))
            print(json.dumps({k:v for k,v in report.items() if k!='events'}), flush=True)

if __name__ == '__main__':
    main()
