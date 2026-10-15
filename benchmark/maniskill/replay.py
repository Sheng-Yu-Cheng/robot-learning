"""Replay local CPU demonstrations into videos and state/action training data."""

import argparse
import copy
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import h5py
import imageio_ffmpeg
import numpy as np

# Resolve the pinned source after moving the upstream submodule into its own folder.
BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parents[1]
OUTPUTS_DIR = BENCHMARK_DIR / "outputs"
sys.path.insert(0, str(BENCHMARK_DIR / "maniskill"))

import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode


def verify_export(output_dir, expected_count):
    """Check every video's frame count/duration against its saved trajectory."""
    metadata = json.loads((output_dir / 'trajectory.json').read_text())
    records = []
    with h5py.File(output_dir / 'trajectory.h5', 'r') as data:
        if len(metadata['episodes']) != expected_count or len(data) != expected_count:
            raise RuntimeError('Saved trajectory count differs from requested count')
        for episode in metadata['episodes']:
            episode_id = episode['episode_id']
            group = data[f'traj_{episode_id}']
            actions, observations = group['actions'], group['obs']
            if observations.shape[0] != actions.shape[0] + 1:
                raise RuntimeError(f'Invalid time alignment in traj_{episode_id}')
            if not np.isfinite(observations[:]).all() or not np.isfinite(actions[:]).all():
                raise RuntimeError(f'Nonfinite training data in traj_{episode_id}')
            if not bool(group['success'][-1]):
                raise RuntimeError(f'Saved trajectory {episode_id} did not succeed')
            video = output_dir / 'videos' / f'{episode_id:06d}.mp4'
            if not video.is_file() or video.stat().st_size == 0:
                raise RuntimeError(f'Missing video: {video}')
            frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
            if frames != actions.shape[0] + 1 or seconds <= 0:
                raise RuntimeError(f'Video/trajectory mismatch: {video}, {frames} frames')
            records.append(dict(episode_id=episode_id, steps=actions.shape[0],
                                observation_dim=observations.shape[-1], action_dim=actions.shape[-1],
                                video=video.relative_to(output_dir).as_posix(), frames=frames, duration_seconds=seconds))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-id', default='PickCube-v1', help='Task ID used as the output namespace')
    parser.add_argument('--source', default='motionplanning', help='Collection under the task raw/ folder')
    parser.add_argument('--traj-path', type=Path, help='Override the input HDF5 file')
    parser.add_argument('--count', type=int, default=50, help='Number of successful demonstrations to save')
    parser.add_argument('--run-name', help='New batch name; default includes count and timestamp')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    for label, value in (('--env-id', args.env_id), ('--source', args.source), ('--run-name', args.run_name)):
        if value is not None and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) is None:
            parser.error(f'{label} must be a single safe directory name')
    source = (args.traj_path or OUTPUTS_DIR / args.env_id / 'raw' / args.source / 'trajectory.h5').resolve()
    if not source.is_file() or not source.with_suffix('.json').is_file():
        parser.error(f'Missing input .h5 or companion .json: {source}')
    metadata = json.loads(source.with_suffix('.json').read_text())
    if metadata['env_info']['env_id'] != args.env_id:
        parser.error('--env-id does not match the task in the input metadata')
    episodes = metadata['episodes']
    if len(episodes) < args.count:
        parser.error(f'Only {len(episodes)} source episodes, fewer than --count')
    env_kwargs = copy.deepcopy(metadata['env_info']['env_kwargs'])
    if env_kwargs.get('sim_backend') not in (None, 'auto', 'cpu', 'physx_cpu'):
        parser.error('Use CPU-collected source demonstrations; CUDA-to-CPU replay is not equivalent')
    modes = {episode['control_mode'] for episode in episodes}
    if len(modes) != 1:
        parser.error('Source must use one control mode')
    env_kwargs.update(obs_mode='state', sim_backend='cpu', render_backend='cpu',
                      render_mode='rgb_array', num_envs=1, control_mode=modes.pop())
    # Some expert demonstrations exceed the task's default 50-step time limit.
    env_kwargs['max_episode_steps'] = max(episode['elapsed_steps'] for episode in episodes) + 1
    run_name = args.run_name or f"state_{env_kwargs['control_mode']}_{args.count}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output_dir = OUTPUTS_DIR / args.env_id / 'replays' / args.source / run_name
    if output_dir.exists():
        parser.error(f'Output already exists; choose a new --run-name: {output_dir}')
    output_dir.mkdir(parents=True)
    (output_dir / 'videos').mkdir()
    source_reference = source.relative_to(PROJECT_ROOT).as_posix() if source.is_relative_to(PROJECT_ROOT) else str(source)
    manifest = dict(env_id=args.env_id, source_collection=args.source, run_name=run_name, source=source_reference, requested=args.count, saved=0, attempted=0,
                    source_type=metadata.get('source_type'), source_commit=metadata.get('commit_info'),
                    observation_mode='state', control_mode=env_kwargs['control_mode'],
                    sim_backend='physx_cpu', render_backend='cpu', episodes=[])
    env = gym.make(metadata['env_info']['env_id'], **env_kwargs)
    recorder = RecordEpisode(
        env, str(output_dir), trajectory_name='trajectory', save_trajectory=True,
        save_video=True, save_on_reset=False, info_on_video=False,
        record_reward=True, record_env_state=True, video_fps=env.unwrapped.control_freq,
        source_type=metadata.get('source_type'), source_desc=f'Replayed from {source_reference}',
    )
    try:
        with h5py.File(source, 'r') as trajectories:
            for episode in episodes:
                reset_kwargs = copy.deepcopy(episode.get('reset_kwargs', {}))
                if 'seed' not in reset_kwargs:
                    if 'episode_seed' not in episode:
                        raise RuntimeError('CPU source episode has no reproducible reset seed')
                    reset_kwargs['seed'] = episode['episode_seed']
                recorder.reset(**reset_kwargs)
                source_id = episode['episode_id']
                actions = trajectories[f'traj_{source_id}']['actions'][:]
                info = {}
                for action in actions:
                    _, _, _, _, info = recorder.step(action)
                manifest['attempted'] += 1
                success = bool(info.get('success', False))
                if success:
                    saved_id = manifest['saved']
                    recorder.flush_trajectory(verbose=False)
                    video_name = f'{saved_id:06d}.mp4'
                    recorder.flush_video(name=Path(video_name).stem, verbose=False)
                    (output_dir / video_name).rename(output_dir / 'videos' / video_name)
                    manifest['episodes'].append(dict(episode_id=saved_id, source_episode_id=source_id,
                                                      steps=len(actions), video=f'videos/{video_name}'))
                    manifest['saved'] += 1
                    print(f"Saved {manifest['saved']}/{args.count}: source episode {source_id}, {len(actions)} steps", flush=True)
                else:
                    recorder.flush_trajectory(save=False)
                    recorder.flush_video(save=False)
                    print(f'Skipped source episode {source_id}: replay did not succeed', flush=True)
                (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
                if manifest['saved'] == args.count:
                    break
    finally:
        recorder.close()
    if manifest['saved'] != args.count:
        raise RuntimeError(f"Only {manifest['saved']} successful replays; see {output_dir / 'manifest.json'}")
    records = verify_export(output_dir, args.count)
    (output_dir / 'verification.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f"PASS: {len(records)} paired videos and training trajectories verified in {output_dir}", flush=True)


if __name__ == '__main__':
    main()
