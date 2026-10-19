"""Train a local diffusion/flow policy on replayed ManiSkill state demonstrations."""

import argparse
import copy
import csv
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.policy.action_policy.diffusion_policy import DiffusionPolicy
from models.policy.action_policy.flow_matching_policy import FlowMatchingPolicy
from experiments.maniskill_action_policy.data import ActionWindows, fit_normalization, load_training_episodes, split_episodes

POLICIES = {'diffusion': DiffusionPolicy, 'flow_matching': FlowMatchingPolicy}


def save_checkpoint(checkpoint, path):
    """Publish atomically, tolerating brief Windows readers without losing work."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temporary)
    for attempt in range(30):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=POLICIES, required=True)
    parser.add_argument('--demo-path', type=Path, required=True)
    parser.add_argument('--recovery-path', type=Path)
    parser.add_argument('--observation-clip', type=float)
    parser.add_argument('--action-normalization', choices=['dataset', 'bounds'], default='dataset')
    parser.add_argument('--rollout-every', type=int, default=0, help='Optimizer updates between closed-loop development evaluations')
    parser.add_argument('--rollout-episodes', type=int, default=3)
    parser.add_argument('--rollout-seed', type=int, default=20000)
    parser.add_argument('--rollout-output-dir', type=Path)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--observation-horizon', type=int, default=2)
    parser.add_argument('--prediction-horizon', type=int, default=16)
    parser.add_argument('--action-horizon', type=int, default=8)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--architecture', choices=['unet', 'mlp'], default='unet')
    parser.add_argument('--flow-prediction', choices=['velocity', 'sample'], default='velocity')
    parser.add_argument('--flow-solver', choices=['heun', 'euler'], default='heun')
    parser.add_argument('--temporal-loss-weight', type=float, default=0.)
    parser.add_argument('--observation-noise', type=float, default=0.)
    parser.add_argument('--observation-std-floor', type=float, default=1e-6)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--ema-decay', type=float, default=0.995)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for this training run')
    parser.add_argument('--max-steps', type=int, help='Maximum optimizer update budget')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.observation_horizon, args.prediction_horizon, args.action_horizon) < 1:
        parser.error('epochs, batch size and horizons must be positive')
    if args.observation_noise < 0 or args.temporal_loss_weight < 0 or args.observation_std_floor <= 0:
        parser.error('Noise and loss weight must be nonnegative; std floor must be positive')
    if args.policy != 'flow_matching' and args.temporal_loss_weight:
        parser.error('Temporal loss is currently implemented for flow_matching')
    if args.lr <= 0 or not 0 <= args.ema_decay < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error('Invalid learning rate, EMA decay or max steps')
    if args.observation_clip is not None and args.observation_clip <= 0:
        parser.error('--observation-clip must be positive')
    if args.rollout_every < 0 or args.rollout_episodes < 1 or (args.rollout_every and args.rollout_output_dir is None):
        parser.error('Rollout selection needs a positive episode count and explicit --rollout-output-dir')
    if args.rollout_output_dir is not None and args.rollout_output_dir.exists():
        parser.error('Choose a new --rollout-output-dir')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device))
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_path = args.demo_path.resolve()
    metadata = json.loads(data_path.with_suffix('.json').read_text())
    env_id = metadata['env_info']['env_id']
    run_dir = args.output_dir.resolve()
    run_name = run_dir.name
    episodes, groups = load_training_episodes(data_path, args.recovery_path)
    if args.recovery_path is not None:
        recovery_metadata = json.loads(args.recovery_path.with_suffix('.json').read_text())
        if recovery_metadata['env_info']['env_id'] != env_id or recovery_metadata['env_info']['env_kwargs'] != metadata['env_info']['env_kwargs']:
            parser.error('Recovery environment/controller must match primary data')
    train_ids, validation_ids = split_episodes(episodes, args.validation_fraction, args.seed, groups)
    action_bounds = None
    if args.action_normalization == 'bounds':
        import gymnasium as gym
        import mani_skill.envs
        # Query native controller bounds; this is not an action-generating policy.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == 'cuda' else []):
            environment = gym.make(env_id, **metadata['env_info']['env_kwargs'])
            try:
                action_bounds = (environment.action_space.low, environment.action_space.high)
            finally:
                environment.close()
    primary_train_ids = [key for key in train_ids if not key.startswith('recovery_')]
    normalization = fit_normalization(episodes, primary_train_ids, args.observation_std_floor, action_bounds)
    train_data = ActionWindows(episodes, train_ids, normalization, args.observation_horizon, args.prediction_horizon, args.observation_clip)
    validation_data = ActionWindows(episodes, validation_ids, normalization, args.observation_horizon, args.prediction_horizon, args.observation_clip)
    # Keep workers at zero for Windows and this small in-memory dataset.
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    first_obs, first_actions = next(iter(episodes.values()))
    model_config = dict(observation_dim=first_obs.shape[1], action_dim=first_actions.shape[1],
                        observation_horizon=args.observation_horizon, prediction_horizon=args.prediction_horizon,
                        action_horizon=args.action_horizon, hidden_dim=args.hidden_dim, architecture=args.architecture)
    if args.policy == 'flow_matching':
        model_config.update(temporal_loss_weight=args.temporal_loss_weight, prediction_type=args.flow_prediction, solver=args.flow_solver)
    policy = POLICIES[args.policy](**model_config).to(device)
    ema = copy.deepcopy(policy).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-6)
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, 'demo_path': str(data_path), 'output_dir': str(run_dir), 'device': str(device), 'run_name': run_name,
              'model_config': model_config, 'source_groups': groups, 'train_ids': train_ids, 'validation_ids': validation_ids,
              'env_info': metadata['env_info'], 'torch_version': str(torch.__version__)}
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'{args.policy} on {device}: {len(train_ids)} train / {len(validation_ids)} validation episodes; '
          f'{len(train_data)} / {len(validation_data)} windows', flush=True)
    print(f'Outputs: {run_dir}', flush=True)
    best_loss, steps = float('inf'), 0
    next_rollout = args.rollout_every
    best_rollout_rank = None
    with (run_dir / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'steps', 'train_loss', 'validation_loss'])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            policy.train()
            total, weight = 0., 0
            for batch in train_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                observations = batch['observations'] + args.observation_noise * torch.randn_like(batch['observations'])
                if args.observation_clip is not None:
                    observations = observations.clamp(-args.observation_clip, args.observation_clip)
                loss = policy.compute_loss(observations, batch['actions'], mask=batch['mask'])
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():
                    for ema_parameter, parameter in zip(ema.parameters(), policy.parameters()):
                        ema_parameter.lerp_(parameter, 1 - args.ema_decay)
                valid = batch['mask'].sum().item()
                total += loss.item() * valid
                weight += valid
                steps += 1
                if args.max_steps is not None and steps >= args.max_steps:
                    break
            # Fixed validation noise/times permit comparisons between epochs without changing train RNG.
            validation_total, validation_weight = 0., 0
            cuda_devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
            with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(args.seed + 10000)
                for batch in validation_loader:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    loss = ema.compute_loss(batch['observations'], batch['actions'], mask=batch['mask'])
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite validation loss')
                    valid = batch['mask'].sum().item()
                    validation_total += loss.item() * valid
                    validation_weight += valid
            validation_loss = validation_total / validation_weight
            row = dict(epoch=epoch, steps=steps, train_loss=total / weight, validation_loss=validation_loss)
            writer.writerow(row)
            stream.flush()
            print(f'Epoch {epoch}: train={row["train_loss"]:.5f}, validation={validation_loss:.5f}, steps={steps}', flush=True)
            checkpoint = dict(config=config, epoch=epoch, steps=steps, normalization=normalization,
                              model_state_dict=policy.state_dict(), ema_state_dict=ema.state_dict(),
                              optimizer_state_dict=optimizer.state_dict(), validation_loss=validation_loss)
            save_checkpoint(checkpoint, run_dir / 'last.pt')
            if validation_loss < best_loss:
                best_loss = validation_loss
                save_checkpoint(checkpoint, run_dir / 'best.pt')
            stop = args.max_steps is not None and steps >= args.max_steps
            if args.rollout_every and (steps >= next_rollout or stop or epoch == args.epochs):
                rollout_dir = args.rollout_output_dir.resolve() / f'step_{steps:06d}'
                subprocess.run([sys.executable, '-B', '-s', '-m', 'experiments.maniskill_action_policy.evaluate',
                                '--checkpoint', str(run_dir / 'last.pt'), '--output-dir', str(rollout_dir),
                                '--episodes', str(args.rollout_episodes), '--seed', str(args.rollout_seed)], check=True)
                metrics = json.loads((rollout_dir / 'metrics.json').read_text())
                rank = (metrics['success_at_end_rate'], metrics['success_once_rate'],
                        sum(e['reward'] for e in metrics['episodes']) / len(metrics['episodes']))
                if best_rollout_rank is None or rank > best_rollout_rank:
                    best_rollout_rank = rank
                    selected = {**checkpoint, 'rollout_validation': dict(metrics=str(rollout_dir / 'metrics.json'),
                                rank=list(rank), seed=args.rollout_seed, episodes=args.rollout_episodes)}
                    save_checkpoint(selected, run_dir / 'best_rollout.pt')
                print(f'Closed-loop development rank: {rank}; best: {best_rollout_rank}', flush=True)
                while next_rollout <= steps:
                    next_rollout += args.rollout_every
            if stop:
                break
    print(f'Finished. Best EMA validation loss: {best_loss:.5f}; checkpoint: {run_dir / "best.pt"}', flush=True)


if __name__ == '__main__':
    main()
