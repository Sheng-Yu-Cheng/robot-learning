"""Replay local CPU demonstrations into videos and state/action training data."""

import argparse
import copy
import json
import re
from pathlib import Path

import gymnasium as gym
import h5py
import imageio_ffmpeg
import numpy as np

import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.utils.actions.conversion import qpos_to_pd_joint_delta_pos


def joint_target_to_tcp_delta(arm, joint_target):
    """FK of absolute arm targets, relative to the actual TCP in robot-root axes."""
    if arm.config.frame != 'root_translation' or arm.config.use_target:
        raise ValueError('TCP conversion requires current-pose root_translation control')
    kinematics = arm.kinematics
    qpos = arm.articulation.get_qpos()[0].cpu().numpy().copy()
    indices = arm.active_joint_indices.cpu().numpy()
    if np.asarray(joint_target).shape != indices.shape:
        raise ValueError('Source arm target dimension differs from controlled joints')
    qpos[indices] = joint_target
    kinematics.pmodel.compute_forward_kinematics(qpos[kinematics.pmodel_active_joint_indices.cpu().numpy()])
    target = kinematics.pmodel.get_link_pose(kinematics.end_link_idx).p
    delta = target - arm.ee_pose_at_base.p[0].cpu().numpy()
    low = arm.action_space_low.cpu().numpy().reshape(-1)
    high = arm.action_space_high.cpu().numpy().reshape(-1)
    return 2 * (delta - low) / (high - low) - 1


def verify_export(output_dir, expected_count, video_count=None):
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
            if 'executed_actions' in group:
                behavior = group['executed_actions'][:]
                if behavior.shape != actions.shape or not np.isfinite(behavior).all():
                    raise RuntimeError('Invalid executed behavior commands')
            if not bool(group['success'][-1]):
                raise RuntimeError(f'Saved trajectory {episode_id} did not succeed')
            if video_count is not None and episode_id >= video_count:
                records.append(dict(episode_id=episode_id, steps=actions.shape[0], observation_dim=observations.shape[-1], action_dim=actions.shape[-1], video=None))
                continue
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
    parser.add_argument('--source', default='motionplanning', help='Source collection label')
    parser.add_argument('--traj-path', type=Path, required=True, help='Input HDF5 with its companion JSON')
    parser.add_argument('--control-mode', choices=['pd_joint_pos', 'pd_joint_delta_pos', 'pd_ee_delta_pos'], help='Convert absolute joint references to joint delta or TCP position delta')
    parser.add_argument('--video-count', type=int, help='Render and record only the first N saved demonstrations; default records all')
    parser.add_argument('--hold-steps', type=int, default=0, help='Physically replay the final absolute joint target for extra settling observations')
    parser.add_argument('--shuffle-seed', type=int, help='Shuffle source episodes before replay')
    parser.add_argument('--source-manifest', type=Path, help='Restrict source IDs/order to a previous replay manifest')
    parser.add_argument('--perturbation-std', type=float, default=0., help='Gaussian behavior noise in normalized target-controller command units')
    parser.add_argument('--perturbation-probability', type=float, default=0.15)
    parser.add_argument('--perturbation-seed', type=int, default=123)
    parser.add_argument('--count', type=int, default=50, help='Number of successful demonstrations to save')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for exported data and videos')
    args = parser.parse_args()
    if args.perturbation_std < 0 or not 0 <= args.perturbation_probability <= 1:
        parser.error('Invalid perturbation magnitude or probability')
    if args.video_count is not None and args.video_count < 0:
        parser.error('--video-count must be nonnegative')
    if args.hold_steps < 0:
        parser.error('--hold-steps must be nonnegative')
    if args.count < 1:
        parser.error('--count must be positive')
    for label, value in (('--env-id', args.env_id), ('--source', args.source)):
        if value is not None and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) is None:
            parser.error(f'{label} must be a single safe directory name')
    source = args.traj_path.resolve()
    if not source.is_file() or not source.with_suffix('.json').is_file():
        parser.error(f'Missing input .h5 or companion .json: {source}')
    metadata = json.loads(source.with_suffix('.json').read_text())
    if metadata.get('action_semantics') == 'expert_labels':
        parser.error('Recovery actions are supervision labels; use original references for new replay')
    if metadata['env_info']['env_id'] != args.env_id:
        parser.error('--env-id does not match the task in the input metadata')
    episodes = list(metadata['episodes'])
    if args.shuffle_seed is not None:
        np.random.default_rng(args.shuffle_seed).shuffle(episodes)
    if args.source_manifest is not None:
        previous = json.loads(args.source_manifest.read_text())
        if Path(previous['source']).resolve() != source:
            parser.error('Source manifest refers to a different reference HDF5')
        by_id = {episode['episode_id']: episode for episode in episodes}
        episodes = [by_id[item['source_episode_id']] for item in previous['episodes']]
    if len(episodes) < args.count:
        parser.error(f'Only {len(episodes)} source episodes, fewer than --count')
    env_kwargs = copy.deepcopy(metadata['env_info']['env_kwargs'])
    if env_kwargs.get('sim_backend') not in (None, 'auto', 'cpu', 'physx_cpu'):
        parser.error('Use CPU-collected source demonstrations; CUDA-to-CPU replay is not equivalent')
    modes = {episode['control_mode'] for episode in episodes}
    if len(modes) != 1:
        parser.error('Source must use one control mode')
    if args.hold_steps and modes != {'pd_joint_pos'}:
        parser.error('--hold-steps requires absolute pd_joint_pos targets; repeating delta actions would move the arm')
    source_mode = next(iter(modes))
    target_mode = args.control_mode or source_mode
    if target_mode != source_mode and not (source_mode == 'pd_joint_pos' and target_mode in {'pd_joint_delta_pos', 'pd_ee_delta_pos'}):
        parser.error('Conversion requires absolute joint position source references')
    if args.perturbation_std and not (source_mode == 'pd_joint_pos' and target_mode in {'pd_joint_delta_pos', 'pd_ee_delta_pos'}):
        parser.error('Recovery collection requires absolute references and a supported delta controller')
    env_kwargs.update(obs_mode='state', sim_backend='cpu', render_backend='cpu',
                      render_mode='rgb_array', num_envs=1, control_mode=target_mode, reconfiguration_freq=1)
    # Some expert demonstrations exceed the task's default 50-step time limit.
    env_kwargs['max_episode_steps'] = (max(episode['elapsed_steps'] for episode in episodes) + args.hold_steps) * (4 if target_mode == 'pd_ee_delta_pos' else 1) + 1
    output_dir = args.output_dir.resolve()
    run_name = output_dir.name
    if output_dir.exists():
        parser.error(f'Output already exists; choose a new --output-dir: {output_dir}')
    output_dir.mkdir(parents=True)
    (output_dir / 'videos').mkdir()
    source_reference = str(source)
    manifest = dict(env_id=args.env_id, source_collection=args.source, run_name=run_name, source=source_reference, requested=args.count, saved=0, attempted=0,
                    source_type=metadata.get('source_type'), source_commit=metadata.get('commit_info'),
                    observation_mode='state', control_mode=env_kwargs['control_mode'],
                    sim_backend='physx_cpu', render_backend='cpu', hold_steps=args.hold_steps, shuffle_seed=args.shuffle_seed, video_count=args.video_count, source_control_mode=source_mode, reset_strategy='reconfigure_each_episode', perturbation_std=args.perturbation_std,
                    perturbation_probability=args.perturbation_probability, perturbation_seed=args.perturbation_seed,
                    action_semantics='expert_labels' if args.perturbation_std else 'executed_commands', episodes=[])
    env = gym.make(metadata['env_info']['env_id'], **env_kwargs)
    manifest['action_dim'] = int(env.action_space.shape[-1])
    if target_mode == 'pd_ee_delta_pos':
        arm = env.unwrapped.agent.controller.controllers['arm']
        manifest['tcp_action'] = dict(frame=arm.config.frame, use_target=arm.config.use_target,
            position_lower=np.asarray(arm.config.pos_lower).tolist(),
            position_upper=np.asarray(arm.config.pos_upper).tolist(),
            components=['delta_x', 'delta_y', 'delta_z', 'gripper'], rotation_control=False,
            conversion='FK of absolute source arm target minus actual current TCP', max_substeps=4)
    recorder = RecordEpisode(
        env, str(output_dir), trajectory_name='trajectory', save_trajectory=True,
        save_video=True, save_video_trigger=lambda _: args.video_count is None or manifest['saved'] < args.video_count,
        save_on_reset=False, info_on_video=False,
        record_reward=True, record_env_state=True, video_fps=env.unwrapped.control_freq,
        source_type=metadata.get('source_type'), source_desc=f'Replayed from {source_reference}',
    )
    expert_labels = {}
    perturbation_rng = np.random.default_rng(args.perturbation_seed)
    try:
        with h5py.File(source, 'r') as trajectories:
            for episode in episodes:
                reset_kwargs = copy.deepcopy(episode.get('reset_kwargs', {}))
                if 'seed' not in reset_kwargs:
                    if 'episode_seed' not in episode:
                        raise RuntimeError('CPU source episode has no reproducible reset seed')
                    reset_kwargs['seed'] = episode['episode_seed']
                _, reset_info = recorder.reset(**reset_kwargs)
                if not reset_info.get('reconfigure', False):
                    raise RuntimeError('CPU replay reset must reconstruct the scene to avoid stale contacts')
                source_id = episode['episode_id']
                actions = trajectories[f'traj_{source_id}']['actions'][:]
                info = {}
                if args.hold_steps:
                    actions = np.concatenate((actions, np.repeat(actions[-1:], args.hold_steps, axis=0)))
                labels = []
                perturbations = 0
                for step_index, action in enumerate(actions):
                    # TCP targets exceeding native bounds get up to four physical steps,
                    # matching the upstream conversion strategy. Every step is recorded.
                    for _ in range(4 if target_mode == 'pd_ee_delta_pos' else 1):
                        command = action
                        saturated = False
                        if target_mode != source_mode:
                            arm = env.unwrapped.agent.controller.controllers['arm']
                            if target_mode == 'pd_ee_delta_pos':
                                width = len(arm.active_joint_indices)
                                delta = joint_target_to_tcp_delta(arm, action[:width])
                            else:
                                width = arm.single_action_space.shape[0]
                                delta = qpos_to_pd_joint_delta_pos(arm, action[:width])
                            saturated = bool((np.abs(delta) > 1).any())
                            command = np.concatenate((np.clip(delta, -1, 1), action[width:]))
                        labels.append(np.asarray(command, dtype=np.float32).copy())
                        # Supervision uses the actual disturbed state, without behavior noise.
                        if (args.perturbation_std and step_index < len(actions) - args.hold_steps
                                and perturbation_rng.random() < args.perturbation_probability):
                            command = np.clip(command + perturbation_rng.normal(0., args.perturbation_std, command.shape),
                                              env.action_space.low.reshape(-1), env.action_space.high.reshape(-1)).astype(np.float32)
                            perturbations += 1
                        _, _, _, _, info = recorder.step(command)
                        if not saturated:
                            break
                manifest['attempted'] += 1
                success = bool(info.get('success', False))
                if success:
                    saved_id = manifest['saved']
                    if args.perturbation_std:
                        expert_labels[saved_id] = np.stack(labels)
                    recorder.flush_trajectory(verbose=False)
                    video_name = f'{saved_id:06d}.mp4'
                    has_video = args.video_count is None or saved_id < args.video_count
                    if has_video:
                        recorder.flush_video(name=Path(video_name).stem, verbose=False)
                        (output_dir / video_name).rename(output_dir / 'videos' / video_name)
                    manifest['episodes'].append(dict(episode_id=saved_id, source_episode_id=source_id,
                                                      steps=len(labels), perturbations=perturbations, reset_reconfigured=bool(reset_info['reconfigure']),
                                                      video=f'videos/{video_name}' if has_video else None))
                    manifest['saved'] += 1
                    print(f"Saved {manifest['saved']}/{args.count}: source episode {source_id}, {len(labels)} steps", flush=True)
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
    if args.perturbation_std:
        with h5py.File(output_dir / 'trajectory.h5', 'r+') as output:
            for episode_id, labels in expert_labels.items():
                group = output[f'traj_{episode_id}']
                group.move('actions', 'executed_actions')
                group.create_dataset('actions', data=labels)
        exported = json.loads((output_dir / 'trajectory.json').read_text())
        exported['action_semantics'] = 'expert_labels'
        exported['executed_action_dataset'] = 'executed_actions'
        (output_dir / 'trajectory.json').write_text(json.dumps(exported, indent=2), encoding='utf-8')
    records = verify_export(output_dir, args.count, args.video_count)
    (output_dir / 'verification.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f"PASS: {len(records)} training trajectories and requested videos verified in {output_dir}", flush=True)


if __name__ == '__main__':
    main()
