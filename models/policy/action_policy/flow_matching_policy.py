"""Conditional flow matching on a straight Gaussian-to-action path."""

import torch
from torch import nn

from .common import initial_sample, masked_mse, observation_condition, validate_actions, validate_config
from .networks import ConditionalMLP, ConditionalUnet1D


class FlowMatchingPolicy(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, *, observation_horizon: int = 2,
                 prediction_horizon: int = 16, action_horizon: int = 8, hidden_dim: int = 64,
                 num_inference_steps: int = 20, solver: str = "heun", clip_sample: bool = True,
                 architecture: str = "unet", temporal_loss_weight: float = 0., prediction_type: str = "velocity"):
        super().__init__()
        validate_config(observation_dim, action_dim, observation_horizon, prediction_horizon, action_horizon)
        if not isinstance(num_inference_steps, int) or num_inference_steps < 1:
            raise ValueError("num_inference_steps must be a positive integer")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.observation_horizon = observation_horizon
        self.prediction_horizon = prediction_horizon
        self.action_horizon = action_horizon
        self.num_inference_steps = num_inference_steps
        if temporal_loss_weight < 0:
            raise ValueError("temporal_loss_weight must be nonnegative")
        if prediction_type not in {'velocity', 'sample'}:
            raise ValueError("prediction_type must be 'velocity' or 'sample'")
        if prediction_type == 'sample' and solver != 'euler':
            raise ValueError("sample prediction currently requires the endpoint-safe Euler solver")
        self.prediction_type = prediction_type
        self.temporal_loss_weight = temporal_loss_weight
        self.solver = solver
        self.clip_sample = clip_sample
        if architecture == 'unet':
            self.network = ConditionalUnet1D(action_dim, observation_dim * observation_horizon, hidden_dim)
        elif architecture == 'mlp':
            self.network = ConditionalMLP(action_dim, observation_dim * observation_horizon, prediction_horizon, hidden_dim)
        else:
            raise ValueError("architecture must be 'unet' or 'mlp'")

    def compute_loss(self, observations, actions, *, mask=None, noise=None, times=None):
        condition = observation_condition(self, observations)
        validate_actions(self, actions, observations)
        if noise is None:
            noise = torch.randn_like(actions)
        else:
            validate_actions(self, noise, observations)
        if times is None:
            times = torch.rand(actions.shape[0], device=actions.device, dtype=actions.dtype)
        if times.shape != (actions.shape[0],) or times.device != actions.device or times.dtype != actions.dtype:
            raise ValueError("times must be floating [B] matching the action device and dtype")
        if not torch.isfinite(times).all() or (times < 0).any() or (times > 1).any():
            raise ValueError("times must be finite and between 0 and 1")
        t = times[:, None, None]
        path = (1 - t) * noise + t * actions
        prediction = self.network(path, times, condition)
        if self.prediction_type == 'sample':
            # x1 prediction is flow matching weighted by (1-t)^2. It avoids
            # making precision of the action depend on cancellation of noise.
            clean = prediction
            loss = masked_mse(clean, actions, mask)
        else:
            clean = path + (1 - t) * prediction
            loss = masked_mse(prediction, actions - noise, mask)
        if self.temporal_loss_weight and self.prediction_horizon > 1:
            # Match demonstrated changes, rather than penalizing all motion.
            pair_mask = None if mask is None else mask[:, 1:] & mask[:, :-1]
            if pair_mask is None or pair_mask.any():
                loss = loss + self.temporal_loss_weight * masked_mse(
                    clean[:, 1:] - clean[:, :-1], actions[:, 1:] - actions[:, :-1], pair_mask)
        return loss

    @torch.no_grad()
    def sample(self, observations, *, num_steps=None, generator=None, initial_noise=None):
        """Integrate t=0 (Gaussian) to t=1 (action); return [B, H, A]."""
        condition = observation_condition(self, observations)
        steps = self.num_inference_steps if num_steps is None else num_steps
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("num_steps must be a positive integer")
        sample = initial_sample(self, observations, initial_noise, generator)
        dt = 1 / steps
        for step in range(steps):
            time = observations.new_full((observations.shape[0],), step / steps)
            prediction = self.network(sample, time, condition)
            if self.prediction_type == 'sample':
                # Never evaluate a singular velocity at t=1. On the last
                # Euler step dt/(1-t)=1, so the output is the predicted x1.
                sample = sample + (prediction - sample) / (steps - step)
                continue
            velocity = prediction
            if self.solver == "heun":
                next_time = observations.new_full((observations.shape[0],), (step + 1) / steps)
                next_velocity = self.network(sample + dt * velocity, next_time, condition)
                sample = sample + 0.5 * dt * (velocity + next_velocity)
            else:
                sample = sample + dt * velocity
        return sample.clamp(-1, 1) if self.clip_sample else sample

    @torch.no_grad()
    def predict_action(self, observations, **sample_kwargs):
        return self.sample(observations, **sample_kwargs)[:, :self.action_horizon]
