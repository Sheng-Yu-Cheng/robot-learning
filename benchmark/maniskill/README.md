# ManiSkill

This directory owns the ManiSkill integration, data preparation, and local
outputs. The upstream Git submodule is `maniskill/`; keep generated files out
of that source tree. The pinned upstream commit is unchanged.

## Directory layout

```text
benchmark/maniskill/
├── README.md
├── replay.py
├── maniskill/                         # upstream Git submodule
└── outputs/                           # local artifacts, ignored by Git
    └── <TaskID>/                      # e.g. PickCube-v1, StackCube-v1
        ├── raw/
        │   └── <source>/              # motionplanning, rl, teleop, ...
        └── replays/
            └── <source>/
                └── <run-name>/
                    ├── trajectory.h5
                    ├── trajectory.json
                    ├── manifest.json
                    ├── verification.json
                    └── videos/
                        └── 000000.mp4 ...
```

Task IDs use the exact ManiSkill environment name. Keep downloaded collections
under `raw/` and derived batches under `replays/`. Within each run, videos are
separate from the training data and reports. Add new task folders using this
same layout. Future training and evaluation results should use sibling categories
such as `training/<run-name>/` and `evaluation/<run-name>/` under their task;
create them only when needed. Do not create a project-root output directory.

## Existing data

`outputs/PickCube-v1/raw/` contains the downloaded `motionplanning`, `rl`, and
`teleop` collections, including their original HDF5/JSON files, sample videos,
and any supplied policy checkpoints. Downloaded source collections stay intact.

The motion-planning collection has 1,000 episodes using `pd_joint_pos` (8D).
The teleoperation collection has only 10 episodes. RL demonstrations were
collected with GPU physics; the local replay tool requires CPU-compatible input.

Prepared batches are at:

- `outputs/PickCube-v1/replays/motionplanning/demos_50/`: 50 successful demonstrations, 3,760 transitions.
- `outputs/PickCube-v1/replays/motionplanning/preview_1/`: the original one-episode preview.

Each exported `obs` has `T+1` state vectors (42D for this PickCube setup), while
`actions` has `T` rows (8D). `manifest.json` maps saved episode IDs to the source
IDs; its source path is relative to the project root. Video paths in both reports
are relative to the run directory. Source `.h5` and `.json` must stay together.

## Download additional tasks

From the repository root, using the existing `maniskill` environment:

```sh
conda activate maniskill
python -m mani_skill.utils.download_demo
python -m mani_skill.utils.download_demo StackCube-v1 --output_dir benchmark/maniskill/outputs/_incoming
```

The first command lists available task IDs. Archives contain their own task/source
directories. After download, move the extracted `<TaskID>/<source>/` collections
into `outputs/<TaskID>/raw/<source>/` and remove the empty staging directories.
Keep downloaded examples and checkpoints with their source collection. Never
mix an archive's raw trajectories into a replay batch.

## Export videos and training data

Run from the repository root, or pass the script's absolute path from another
working directory. Default input/output locations resolve relative to this file,
not the shell's working directory.

```sh
conda activate maniskill
python benchmark/maniskill/replay.py --env-id PickCube-v1 --source motionplanning --count 50
```

The default batch name contains the observation mode, controller, count, and a
timestamp, so repeated runs do not collide. To choose a descriptive batch name:

```sh
python benchmark/maniskill/replay.py --env-id PickCube-v1 --source motionplanning --count 1 --run-name preview_next
python benchmark/maniskill/replay.py --env-id StackCube-v1 --source motionplanning --count 50 --run-name baseline_50
```

The second command requires StackCube data to have been downloaded and placed in
its `raw/motionplanning/` folder first. `--traj-path` can select another input
HDF5 file; its task ID must match `--env-id`. An existing run name is rejected.
All outputs remain under this integration's `outputs/<TaskID>/replays/` directory.

The script replays original actions and saves the requested number of successful
episodes, skipping failures. It preserves the controller, expands the time limit
to fit expert trajectories, and records state observations, actions, rewards,
environment states, metadata, and one MP4 per episode. This reuses existing
expert data; it does not collect new human demonstrations or train a policy.

On Windows use one CPU simulation environment and CPU image readback. All
rendering is headless and saved as video. The script imports the pinned nested
submodule directly, so a previous editable installation pointing at the old
folder does not need to be modified to run this wrapper. Other direct
`python -m mani_skill...` commands require the environment's editable install
to point at `benchmark/maniskill/maniskill/`.

Checks verify successful final states, finite training arrays, `T+1` observations,
video frame counts, and positive video duration. Every run has its own reports.

## Training preparation

Raw downloads may omit observations; simulator state is not a replacement for
policy observations. Replaying with `replay.py` produces state observations.
Use `obs[:-1]` paired with `actions` for single steps, or build windows using
`obs[t-O+1:t+1]` and `actions[t:t+H]` for the local action policies. Split by whole
episode before windowing, fit normalization on training episodes only, and mask
padded future actions. Preserve the controller configuration for evaluation.

The original joint-position actions are not normalized end-effector deltas;
normalize them before using the local policy's default clipped output.
See [action policy models](../../models/policy/action_policy/README.md).

## References

- [Local imitation-learning setup](maniskill/docs/source/user_guide/learning_from_demos/setup.md)
- [Local trajectory replay documentation](maniskill/docs/source/user_guide/datasets/replay.md)
- [Bundled Diffusion Policy baseline](maniskill/examples/baselines/diffusion_policy/README.md)
- [Official demonstration dataset](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations)
