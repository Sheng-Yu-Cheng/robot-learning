"""Episode-level splitting, training-only normalization, and action windows."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def load_episodes(path):
    episodes = {}
    dimensions = None
    with h5py.File(path, 'r') as data:
        for key in sorted(data, key=lambda key: int(key.split('_')[-1])):
            group = data[key]
            if 'obs' not in group or not isinstance(group['obs'], h5py.Dataset):
                raise ValueError('Replay the data with obs_mode=state first')
            obs = np.asarray(group['obs'], dtype=np.float32)
            actions = np.asarray(group['actions'], dtype=np.float32)
            if obs.ndim != 2 or actions.ndim != 2 or len(actions) < 1 or len(obs) != len(actions) + 1:
                raise ValueError(f'{key}: expected obs [T+1,D] and actions [T,A]')
            if not np.isfinite(obs).all() or not np.isfinite(actions).all():
                raise ValueError(f'{key}: nonfinite input data')
            shape = (obs.shape[1], actions.shape[1])
            if dimensions is not None and dimensions != shape:
                raise ValueError('Episode dimensions differ')
            dimensions = shape
            episodes[key] = (obs, actions)
    if len(episodes) < 2:
        raise ValueError('Need at least two episodes for separate train/validation sets')
    return episodes


def split_episodes(episodes, validation_fraction, seed, groups=None):
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    if groups is not None:
        grouped = {}
        for key in episodes:
            grouped.setdefault(groups[key], []).append(key)
        if len(grouped) < 2:
            raise ValueError('Need at least two independent source groups')
        group_ids = list(grouped)
        np.random.default_rng(seed).shuffle(group_ids)
        n = min(len(group_ids) - 1, max(1, round(len(group_ids) * validation_fraction)))
        return ([key for group in group_ids[n:] for key in grouped[group]],
                [key for group in group_ids[:n] for key in grouped[group]])
    keys = list(episodes)
    np.random.default_rng(seed).shuffle(keys)
    validation_count = min(len(keys) - 1, max(1, round(len(keys) * validation_fraction)))
    return keys[validation_count:], keys[:validation_count]


def fit_normalization(episodes, train_ids, observation_std_floor=1e-6, action_bounds=None):
    obs = np.concatenate([episodes[key][0][:-1] for key in train_ids])
    actions = np.concatenate([episodes[key][1] for key in train_ids])
    mean = obs.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = obs.std(axis=0, dtype=np.float64).astype(np.float32)
    if observation_std_floor <= 0:
        raise ValueError("observation_std_floor must be positive")
    if observation_std_floor <= 1e-6:
        # Preserve the original constant-feature behavior at the legacy floor.
        std = np.where(std < 1e-6, 1., np.maximum(std, observation_std_floor))
    else:
        std = np.maximum(std, observation_std_floor)
    low, high = actions.min(axis=0), actions.max(axis=0)
    if action_bounds is not None:
        low, high = (np.asarray(x, dtype=np.float32).reshape(-1) for x in action_bounds)
        if low.shape != actions.shape[1:] or high.shape != low.shape or not np.isfinite(low).all() or not np.isfinite(high).all() or (high <= low).any():
            raise ValueError('Action bounds must be finite and match the action dimension')
    return {key: torch.from_numpy(value.astype(np.float32)) for key, value in dict(
        observation_mean=mean, observation_std=std, action_center=(high + low) / 2,
        action_scale=np.maximum((high - low) / 2, 1e-6)).items()}


class ActionWindows(Dataset):
    def __init__(self, episodes, ids, normalization, observation_horizon, prediction_horizon, observation_clip=None):
        self.observation_clip = observation_clip
        self.episodes = episodes
        self.normalization = normalization
        self.observation_horizon = observation_horizon
        self.prediction_horizon = prediction_horizon
        self.indices = [(key, t) for key in ids for t in range(len(episodes[key][1]))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        key, t = self.indices[index]
        obs, actions = self.episodes[key]
        history = np.arange(t - self.observation_horizon + 1, t + 1).clip(0)
        future = np.arange(t, t + self.prediction_horizon)
        observation = torch.from_numpy(obs[history])
        action = torch.from_numpy(actions[future.clip(max=len(actions) - 1)])
        stats = self.normalization
        return dict(observations=normalize_observations(observation, stats, self.observation_clip),
                    actions=(action - stats['action_center']) / stats['action_scale'],
                    mask=torch.from_numpy(future < len(actions)))


def normalize_observations(observations, normalization, clip=None):
    """The same transformation for training, validation and online inference."""
    normalized = (observations - normalization['observation_mean']) / normalization['observation_std']
    if clip is not None:
        if clip <= 0:
            raise ValueError('Observation clip must be positive')
        normalized = normalized.clamp(-clip, clip)
    return normalized


def load_training_episodes(primary_path, recovery_path=None):
    """Keep augmented copies of the same source trajectory in one split."""
    primary = load_episodes(primary_path)
    if recovery_path is None:
        return primary, None
    episodes = dict(primary)
    groups = {}
    for path, prefix in [(Path(primary_path), ''), (Path(recovery_path), 'recovery_')]:
        manifest = json.loads((path.parent / 'manifest.json').read_text())
        mapping = {f"traj_{e['episode_id']}": str(Path(manifest['source']).resolve()) + ':' + str(e['source_episode_id'])
                   for e in manifest['episodes']}
        if prefix:
            metadata = json.loads(path.with_suffix('.json').read_text())
            if metadata.get('action_semantics') != 'expert_labels':
                raise ValueError('Recovery file must explicitly declare expert supervision labels')
            if manifest['saved'] != manifest['requested']:
                raise ValueError('Recovery collection is incomplete')
        loaded = primary if not prefix else load_episodes(path)
        for key, value in loaded.items():
            name = prefix + key
            episodes[name] = value
            groups[name] = mapping[key]
    dimensions = {(obs.shape[1], actions.shape[1]) for obs, actions in episodes.values()}
    if len(dimensions) != 1:
        raise ValueError('Primary and recovery data dimensions must match')
    return episodes, groups
