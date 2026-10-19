"""Evaluate an action-policy checkpoint and save headless rollout videos."""

import argparse
from collections import deque
import json
from pathlib import Path

import gymnasium as gym
import imageio_ffmpeg
import numpy as np
import torch

import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode
from experiments.maniskill_action_policy.data import normalize_observations
from models.policy.action_policy.diffusion_policy import DiffusionPolicy
from models.policy.action_policy.flow_matching_policy import FlowMatchingPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--action-horizon', type=int, help='Override executed chunk length, at most the prediction horizon')
    parser.add_argument('--stop-on-success', action='store_true', help='Legacy diagnostic mode; default continues to the fixed horizon after success')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=150)
    parser.add_argument('--inference-steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for evaluation metrics and videos')
    args = parser.parse_args()
    if min(args.episodes, args.max_episode_steps, args.inference_steps) < 1:
        parser.error('Episode count, horizon and inference steps must be positive')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device))
    torch.set_num_threads(4)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    kind = {'diffusion': DiffusionPolicy, 'flow_matching': FlowMatchingPolicy}[config['policy']]
    policy = kind(**config['model_config']).to(device)
    policy.load_state_dict(checkpoint['ema_state_dict'])
    policy.eval()
    if args.action_horizon is not None:
        if not 1 <= args.action_horizon <= policy.prediction_horizon:
            parser.error('--action-horizon must be within the prediction horizon')
        policy.action_horizon = args.action_horizon
    torch.manual_seed(args.seed)
    stats = {key: value.to(device) for key, value in checkpoint['normalization'].items()}
    env_info = config['env_info']
    env_id = env_info['env_id']
    run_dir = args.output_dir.resolve()
    run_name = run_dir.name
    env_kwargs = dict(env_info['env_kwargs'])
    env_kwargs.update(obs_mode='state', num_envs=1, sim_backend='cpu', render_backend='cpu',
                      render_mode='rgb_array', max_episode_steps=args.max_episode_steps, reconfiguration_freq=1)
    run_dir.mkdir(parents=True, exist_ok=False)
    base_env = gym.make(env_id, **env_kwargs)
    env = RecordEpisode(base_env, str(run_dir / 'videos'), save_trajectory=False,
                        save_video=True, save_on_reset=False, video_fps=base_env.unwrapped.control_freq)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    results = []
    try:
        for episode in range(args.episodes):
            observation, reset_info = env.reset(seed=args.seed + episode)
            if not reset_info.get('reconfigure', False):
                raise RuntimeError('CPU evaluation reset must reconstruct the scene to avoid stale contacts')
            initial_flags = {key: bool(value.item()) for key, value in reset_info.items()
                             if key.startswith('is_') and isinstance(value, torch.Tensor)
                             and value.numel() == 1 and value.dtype == torch.bool}
            history = deque([observation[0].clone()] * policy.observation_horizon, maxlen=policy.observation_horizon)
            total_reward, seen_success, done, steps = 0., False, False, 0
            native_flags = {}
            commands = []
            squared_joint_velocities = []
            success_streak, max_success_streak = 0, 0
            while not done:
                condition = torch.stack(list(history)).unsqueeze(0).to(device)
                condition = normalize_observations(condition, stats, config.get('observation_clip'))
                with torch.no_grad():
                    normalized_actions = policy.predict_action(condition, num_steps=args.inference_steps, generator=generator)[0]
                    actions = (normalized_actions * stats['action_scale'] + stats['action_center']).cpu()
                for action in actions:
                    command = np.clip(action.numpy()[None], env.action_space.low, env.action_space.high)
                    commands.append(command[0].copy())
                    observation, reward, terminated, truncated, info = env.step(command)
                    for key, value in info.items():
                        if key.startswith('is_') and isinstance(value, torch.Tensor) and value.numel() == 1 and value.dtype == torch.bool:
                            native_flags.setdefault(key, []).append(bool(value.item()))
                    history.append(observation[0].clone())
                    squared_joint_velocities.append(float(base_env.unwrapped.agent.robot.get_qvel().square().mean().item()))
                    steps += 1
                    total_reward += reward.item()
                    final_success = bool(info.get('success', False))
                    seen_success |= final_success
                    success_streak = success_streak + 1 if final_success else 0
                    max_success_streak = max(max_success_streak, success_streak)
                    done = (bool(truncated.any()) or steps >= args.max_episode_steps
                            or (bool(terminated.any()) and (args.stop_on_success or not final_success)))
                    if done:
                        break
            name = f'{episode:06d}'
            env.flush_video(name=name, verbose=False)
            video = run_dir / 'videos' / f'{name}.mp4'
            frames, duration = imageio_ffmpeg.count_frames_and_secs(str(video))
            if frames != steps + 1 or duration <= 0 or video.stat().st_size == 0:
                raise RuntimeError(f'Invalid rollout video: {video}')
            results.append(dict(episode=episode, seed=args.seed + episode, steps=steps, reward=total_reward,
                                success_once=seen_success, success_at_end=final_success,
                                reset_reconfigured=True, initial_native_flags=initial_flags,
                                video=f'videos/{name}.mp4', frames=frames, duration_seconds=duration,
                                action_delta_rms=float(np.sqrt(np.mean(np.diff(np.asarray(commands), axis=0)**2))) if len(commands) > 1 else 0.,
                                joint_velocity_rms=float(np.sqrt(np.mean(squared_joint_velocities))),
                                max_success_streak=max_success_streak, final_success_streak=success_streak,
                                native_flag_rates={key: sum(values) / len(values) for key, values in native_flags.items()},
                                native_flag_once={key: any(values) for key, values in native_flags.items()},
                                placement_and_static_fraction=float(np.mean(np.asarray(native_flags['is_obj_placed']) & np.asarray(native_flags['is_robot_static'])))
                                if 'is_obj_placed' in native_flags and 'is_robot_static' in native_flags else None))
            print(f'Episode {episode}: success_once={seen_success}, success_at_end={final_success}, reward={total_reward:.3f}', flush=True)
            summary = dict(checkpoint=str(checkpoint_path), policy=config['policy'], training_steps=checkpoint['steps'],
                           action_horizon=policy.action_horizon, seed=args.seed, reset_strategy='reconfigure_each_episode',
                           inference_steps=args.inference_steps, max_episode_steps=args.max_episode_steps,
                           stop_on_success=args.stop_on_success, termination_mode='stop_on_success' if args.stop_on_success else 'continue_after_success',
                           success_once_rate=sum(item['success_once'] for item in results) / len(results),
                           success_at_end_rate=sum(item['success_at_end'] for item in results) / len(results), episodes=results)
            (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    finally:
        env.close()
    print(f"Results and videos: {run_dir}; success_once_rate={summary['success_once_rate']:.1%}", flush=True)


if __name__ == '__main__':
    main()
