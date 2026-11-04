"""Input checks and loss reduction used by both local action policies."""

import torch


def validate_config(observation_dim, action_dim, observation_horizon, prediction_horizon, action_horizon):
    for name, value in (("observation_dim", observation_dim), ("action_dim", action_dim),
                        ("observation_horizon", observation_horizon),
                        ("prediction_horizon", prediction_horizon), ("action_horizon", action_horizon)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if action_horizon > prediction_horizon:
        raise ValueError("action_horizon cannot exceed prediction_horizon")


def observation_condition(policy, observations):
    expected = (policy.observation_horizon, policy.observation_dim)
    if observations.ndim != 3 or observations.shape[0] < 1 or tuple(observations.shape[1:]) != expected:
        raise ValueError(f"observations must have shape [B, {expected[0]}, {expected[1]}] with B > 0")
    parameter = next(policy.parameters())
    if observations.device != parameter.device or observations.dtype != parameter.dtype:
        raise ValueError("observations must match the policy device and floating dtype")
    return observations.flatten(start_dim=1)


def validate_actions(policy, actions, observations):
    shape = (observations.shape[0], policy.prediction_horizon, policy.action_dim)
    if tuple(actions.shape) != shape:
        raise ValueError(f"actions must have shape {shape}")
    if actions.device != observations.device or actions.dtype != observations.dtype:
        raise ValueError("actions and observations must have the same device and dtype")


def masked_mse(prediction, target, mask=None):
    error = (prediction - target).square().mean(dim=-1)
    if mask is None:
        return error.mean()
    if mask.shape != error.shape or mask.dtype != torch.bool or mask.device != error.device:
        raise ValueError("mask must be boolean [B, H] on the same device; True means valid")
    if not mask.any():
        raise ValueError("mask must include at least one valid action")
    return error[mask].mean()


def initial_sample(policy, observations, initial_noise, generator):
    shape = (observations.shape[0], policy.prediction_horizon, policy.action_dim)
    if initial_noise is None:
        return torch.randn(shape, device=observations.device, dtype=observations.dtype, generator=generator)
    validate_actions(policy, initial_noise, observations)
    return initial_noise.clone()
