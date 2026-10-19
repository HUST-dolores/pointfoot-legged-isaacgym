# Branch Guide

Source inventory date: 2026-10-19 (Asia/Shanghai). This file describes code
ownership and artifact locations, not live training status or acceptance results.
Paths below belong to the named branch; not every entry exists on every branch.

## Branches

| Branch | Purpose and status | Entry points | Results and notes |
| --- | --- | --- | --- |
| `codesign` | Committed jump/co-design baseline, supplemented with four matched evaluation protocols and plotting source. | `legged_gym/scripts/train.py`, `run_nested_four_objective_codesign.py`; `payload_step_launch.py`, `terrain_step_launch.py`, `eccentric_payload_launch.py`, `roll_step_response_launch.py` under the same scripts directory. | `logs/wheelfoot_flat/WF_TRON1A/`; `logs/{payload_step_matched_20261016,terrain_step_matched_20261016,eccentric_payload_matched_20261018,roll_step_response_20261018}/`; protocols in `docs/experiments/`. |
| `payload-slide-lab` | Independent sliding-contact validation and bounded residual payload-retention policy. Reward-comparison source is also present; inspect its own status before claiming completion. | `bash play_payload_retention.sh`; modules in `legged_gym/scripts/sliding_lab/`, including `retention_train`, `retention_eval`, and `retention_reward_study`. | `logs/payload_retention_20261018/`; see `PAYLOAD_SLIDING_LAB.md`, `RETENTION_EXPERIMENT.md`, `SLIDING_PROBLEMS.md`. Checkpoints remain external to Git. |
| `ship-motion` | Prescribed moving-deck balance, sensor comparisons, mobility and model-prior experiments. Active research; do not infer every mode is validated. | `legged_gym/scripts/train_ship_motion.py`, `launch_ship_training.py`; `play.py --task wheelfoot_ship` (also `_prior`, `_roll`, `_step`). | `logs/wheelfoot_ship/`; see `docs/ship_motion.md` and each run's `status.md`, including `fleet_headingyaw_ft_20261015/status.md`. |
| `load` | Older payload branch with friction, force/kick and contact diagnostics. Not the learned retention-policy branch. | `legged_gym/scripts/debug_payload_slide_scene.py`; `play.py` options in `docs/play_options_reference.md`. | `logs/wheelfoot_flat/WF_TRON1A/`; diagnostic output depends on the selected command. |
| `jump` | Historical event-triggered jump baseline. Keep as a reference. | `legged_gym/scripts/train.py --task wheelfoot_jump`. | `logs/wheelfoot_jump/`; historical checkpoints are external. |
| `master` | Historical payload-estimation reference, not the current integration baseline. | Historical `legged_gym/scripts/train.py` and `play.py`. | Historical `logs/` and notebooks; do not silently substitute for `codesign`. |

The planned terrain-aware unilateral stepping task does not have a branch yet.
Create a separate branch/worktree from the reviewed `codesign` commit when work
starts. Do not import moving-deck or sliding-contact assumptions by accident.

## Local Worktrees and External Archives

Local worktrees share Git history, but have separate files and staging areas:
`/home/xu/limx_rl/pointfoot-legged-gym` (`codesign`) and sibling directories
`pointfoot-legged-gym-load`, `pointfoot-legged-gym-payload-slide`, and
`pointfoot-legged-gym-ship-motion`.

Long-form experiment handoffs live under `/home/xu/Desktop/codesign_worklog/`.
The completed retention handoff is indexed by
`payload_retention_2026-10-18/final_handoff/` there. Other Desktop paper/figure
and simulation packages remain local archives; their absence from Git is
intentional. Keep independent backups of those artifacts.

## Running and Reproducing

Run from the intended worktree with `ROBOT_TYPE=WF_TRON1A` and `PYTHONPATH="$PWD"`.
Use the Python environment containing the matching Isaac Gym SDK and PyTorch;
an editable install pointing at another worktree can otherwise load wrong code.
The vendor SDK, checkpoints, saved run configurations and archived task snapshots
referenced from `logs/` must be restored separately. A source clone alone does
not reproduce a trained experiment.

Record the exact commit, configuration, seed, checkpoint hash and output directory
for each new run. For old experiments retain their original source hashes and
archives; this later source snapshot does not replace their original provenance.

## Maintenance and Publication

- Use branches for distinct features/research directions, not individual seeds.
- Commit source, configuration, tests and concise protocol/reproduction notes.
- Exclude generated figures, papers, videos, raw trajectories and weights from
  new commits. Ignore rules do not remove files already in Git history.
- Keep feature changes isolated; merge or cherry-pick reviewed shared fixes only
  after appropriate regression tests. Do not force unrelated research together.
- Use tags for reviewed milestones. Before retiring a branch/worktree, preserve
  its committed source and independently archive all ignored artifacts.
- Publish only to `origin` (`HUST-dolores/pointfoot-legged-isaacgym`), not the
  vendor `upstream`. Do not force-push, rewrite history or delete remote branches
  as part of a source backup.
