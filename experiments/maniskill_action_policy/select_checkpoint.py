"""Select frozen checkpoints with full-horizon closed-loop development rollouts."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--max-episode-steps', type=int, default=150)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_episode_steps < 1:
        parser.error('Episodes and horizon must be positive')
    sources = [path.resolve() for path in args.checkpoints]
    if not all(path.is_file() for path in sources):
        parser.error('All frozen checkpoints must exist; run after training finishes')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records, best_rank, selected = [], None, None
    for index, source in enumerate(sources):
        candidate = output / f'candidate_{index:02d}'
        candidate.mkdir()
        snapshot = candidate / 'checkpoint.pt'
        shutil.copyfile(source, snapshot)
        rollout = candidate / 'evaluation'
        subprocess.run([sys.executable, '-B', '-s', '-m', 'experiments.maniskill_action_policy.evaluate',
                        '--checkpoint', str(snapshot), '--output-dir', str(rollout),
                        '--episodes', str(args.episodes), '--seed', str(args.seed),
                        '--max-episode-steps', str(args.max_episode_steps)], check=True)
        metrics = json.loads((rollout / 'metrics.json').read_text())
        if metrics['stop_on_success'] or any(e['steps'] != args.max_episode_steps for e in metrics['episodes']):
            raise RuntimeError('Selection requires full-horizon development rollouts')
        rank = (metrics['success_at_end_rate'], metrics['success_once_rate'],
                sum(e['reward'] for e in metrics['episodes']) / len(metrics['episodes']))
        checkpoint = torch.load(snapshot, map_location='cpu', weights_only=True)
        record = dict(source=str(source), snapshot=str(snapshot), epoch=checkpoint['epoch'],
                      updates=checkpoint['steps'], rank=list(rank), metrics=str(rollout / 'metrics.json'))
        records.append(record)
        if best_rank is None or rank > best_rank:
            best_rank, selected = rank, record
            checkpoint['rollout_validation'] = dict(rank=list(rank), metrics=record['metrics'],
                                                     seed=args.seed, episodes=args.episodes,
                                                     termination_mode='continue_after_success')
            torch.save(checkpoint, output / 'selected.pt')
        (output / 'selection.json').write_text(json.dumps(dict(
            development_seed=args.seed, episodes=args.episodes, max_episode_steps=args.max_episode_steps,
            criterion=['success_at_end_rate', 'success_once_rate', 'mean_reward'],
            candidates=records, selected=selected), indent=2), encoding='utf-8')
    print(f'Selected frozen checkpoint: {selected}; output: {output / "selected.pt"}', flush=True)


if __name__ == '__main__':
    main()
