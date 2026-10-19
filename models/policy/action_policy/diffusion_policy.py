"""Conditional action diffusion: epsilon loss, cosine DDPM schedule, DDIM sampling."""

import math

import torch
from torch import nn

from .common import initial_sample, masked_mse, observation_condition, validate_actions, validate_config
from .networks import ConditionalMLP, ConditionalUnet1D


class DiffusionPolicy(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, *, observation_horizon: int = 2,
                 prediction_horizon: int = 16, action_horizon: int = 8, hidden_dim: int = 64,
                 num_train_steps: int = 100, num_inference_steps: int = 20,
                 clip_sample: bool = True, architecture: str = "unet"):
        super().__init__()
        validate_config(observation_dim, action_dim, observation_horizon, prediction_horizon, action_horizon)
        if not isinstance(num_train_steps, int) or num_train_steps < 2:
            raise ValueError("num_train_steps must be an integer >= 2")
        if not isinstance(num_inference_steps, int) or not 1 <= num_inference_steps <= num_train_steps:
            raise ValueError("num_inference_steps must be between 1 and num_train_steps")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.observation_horizon = observation_horizon
        self.prediction_horizon = prediction_horizon
        self.action_horizon = action_horizon
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.clip_sample = clip_sample
        if architecture == 'unet':
            self.network = ConditionalUnet1D(action_dim, observation_dim * observation_horizon, hidden_dim)
        elif architecture == 'mlp':
            self.network = ConditionalMLP(action_dim, observation_dim * observation_horizon, prediction_horizon, hidden_dim)
        else:
            raise ValueError("architecture must be 'unet' or 'mlp'")
        grid = torch.linspace(0, 1, num_train_steps + 1, dtype=torch.float64)
        cumulative = torch.cos((grid + 0.008) / 1.008 * math.pi / 2).square()
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999)
        self.register_buffer("alpha_bar", (1 - betas).cumprod(dim=0).float())

    def compute_loss(self, observations, actions, *, mask=None, noise=None, timesteps=None):
        """Actions must be normalized; a mask excludes padded target positions."""
        condition = observation_condition(self, observations)
        validate_actions(self, actions, observations)
        batch = actions.shape[0]
        if noise is None:
            noise = torch.randn_like(actions)
        else:
            validate_actions(self, noise, observations)
        if timesteps is None:
            timesteps = torch.randint(self.num_train_steps, (batch,), device=actions.device)
        if (timesteps.shape != (batch,) or timesteps.dtype != torch.long
                or timesteps.device != actions.device):
            raise ValueError("timesteps must be torch.long [B] on the action device")
        if (timesteps < 0).any() or (timesteps >= self.num_train_steps).any():
            raise ValueError("timesteps outside the training schedule")
        alpha = self.alpha_bar[timesteps, None, None]
        noisy = alpha.sqrt() * actions + (1 - alpha).sqrt() * noise
        prediction = self.network(noisy, timesteps.to(actions.dtype) / (self.num_train_steps - 1), condition)
        return masked_mse(prediction, noise, mask)

    @torch.no_grad()
    def sample(self, observations, *, num_steps=None, generator=None, initial_noise=None):
        """Return the whole future [B, prediction_horizon, A] using deterministic DDIM.

        Randomness comes only from the initial Gaussian. Call eval() for inference.
        """
        condition = observation_condition(self, observations)
        steps = self.num_inference_steps if num_steps is None else num_steps
        if not isinstance(steps, int) or not 1 <= steps <= self.num_train_steps:
            raise ValueError("num_steps must be between 1 and num_train_steps")
        sample = initial_sample(self, observations, initial_noise, generator)
        schedule = torch.linspace(self.num_train_steps - 1, 0, steps).round().long().tolist()
        for index, timestep in enumerate(schedule):
            time = observations.new_full((observations.shape[0],), timestep / (self.num_train_steps - 1))
            noise = self.network(sample, time, condition)
            alpha = self.alpha_bar[timestep]
            clean = (sample - (1 - alpha).sqrt() * noise) / alpha.sqrt()
            if self.clip_sample:
                clean = clean.clamp(-1, 1)
                # Recompute epsilon to keep the DDIM direction consistent with clipped x0.
                noise = (sample - alpha.sqrt() * clean) / (1 - alpha).sqrt()
            previous_alpha = self.alpha_bar[schedule[index + 1]] if index + 1 < len(schedule) else sample.new_tensor(1.)
            sample = previous_alpha.sqrt() * clean + (1 - previous_alpha).sqrt() * noise
        return sample

    @torch.no_grad()
    def predict_action(self, observations, **sample_kwargs):
        """First action_horizon future actions; replan after executing this chunk."""
        return self.sample(observations, **sample_kwargs)[:, :self.action_horizon]
