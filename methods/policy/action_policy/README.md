# Action Policies

Two conditional action-chunk generators implemented with PyTorch only:

- `DiffusionPolicy`: DDPM epsilon-prediction training with a squared-cosine noise schedule; deterministic DDIM inference.
- `FlowMatchingPolicy`: conditional flow matching with a straight Gaussian-to-action path; Euler or Heun integration.

Both use a compact temporal 1D U-Net with observation-history conditioning,
sinusoidal time features, GroupNorm, and FiLM residual blocks. The network has
one down/up stage and preserves arbitrary prediction lengths, including odd
lengths. It predicts a whole action sequence rather than independent actions.
These are small implementations of the formulations, not full reproductions of
the cited repositories or pretrained policies. The same local network is reused
because both objectives need a time-conditioned map of action sequences.

## Research references

Read before implementation:

- [Diffusion Policy project and paper](https://diffusion-policy.cs.columbia.edu/).
- [Original low-dimensional policy](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_lowdim_policy.py) and [conditional U-Net](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py): observation-conditioned action diffusion.
- [HRI-EU flow matching policy example](https://github.com/HRI-EU/flow_matching/blob/master/examples/flow_pusht.py): conditional velocity prediction for robot action sequences.
- [Meta flow matching library](https://github.com/facebookresearch/flow_matching): probability paths and ODE-based generation.
- [Local ManiSkill DP baseline](../../../benchmark/maniskill/maniskill/examples/baselines/diffusion_policy/README.md): task-specific data windows and experiment setup.

This implementation is independently written. It uses a Gaussian prior in both
flow training and inference, and integrates the complete interval from 0 to 1.
It does not reproduce the uniform inference initialization seen in the cited
HRI-EU example. No additional repository, package, or submodule is required.

## Tensor and data contract

| Tensor | Shape | Meaning |
| --- | --- | --- |
| observations | `[B, O, D]` | Fixed-length history ending at current time `t` |
| actions | `[B, H, A]` | Targets `a[t], ..., a[t + H - 1]` |
| mask (optional) | `[B, H]`, bool | True for valid action targets |
| sample output | `[B, H, A]` | Whole predicted future |
| predict_action output | `[B, K, A]` | First `K` actions to execute before replanning |

`O=observation_horizon`, `H=prediction_horizon`, `K=action_horizon` with `K <= H`.
All inputs must have the policy's floating dtype and device. The entry accepts
state vectors or upstream encoder features, not raw images. Observation scaling
and action normalization belong to the experiment; fit statistics on training
episodes only and save them with the checkpoint. For the default output clipping,
normalize action targets to `[-1, 1]`, then unnormalize predictions to the
controller's action space. Set `clip_sample=False` if using another action scale.

The action sequence starts at the current time, with no historical action
prefix. If adapting ManiSkill's official DP loader, drop its first `O - 1`
actions and choose a long enough source window to retain `H` future targets.
Pad observation history at episode start; use finite padding and mask padded
future targets at episode end. A mask changes loss reduction, not convolutional
context or the sampler. Windows sampled actions may need `.cpu()` before passing
them to a CPU simulation.

## Training and inference API

Run from the repository root with an existing PyTorch environment such as
`maniskill`. No package installation is needed for these modules.

```python
import torch
from models.policy.action_policy import DiffusionPolicy, FlowMatchingPolicy

policy = DiffusionPolicy(
    observation_dim=32,
    action_dim=7,
    observation_horizon=2,
    prediction_horizon=16,
    action_horizon=8,
).to('cuda')
# Alternatively: FlowMatchingPolicy(..., solver='heun') with the same dimensions.

# obs: normalized [B, 2, 32]; targets: normalized [B, 16, 7].
# These tensors must come from the experiment's dataset and share the policy device.
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
policy.train()
optimizer.zero_grad()
loss = policy.compute_loss(obs, targets, mask=valid_mask)
loss.backward()
optimizer.step()

policy.eval()
# Gaussian sampling is reproducible using a generator on the same device.
generator = torch.Generator(device='cuda').manual_seed(0)
action_chunk = policy.predict_action(obs, num_steps=20, generator=generator)
```

`sample()` returns the full sequence. `predict_action()` returns its first `K`
steps. Both disable gradients. `initial_noise=[B,H,A]` optionally provides an
explicit initial Gaussian sample for controlled comparisons. Diffusion supports
1 through `num_train_steps` inference steps (default 100 training / 20 inference).
Flow accepts any positive inference step count (default 20); Heun evaluates the
network twice per step, Euler once. Lower step counts trade accuracy for speed.

## Objectives

Diffusion uses a random discrete step `k` and Gaussian noise `epsilon`:

```text
x_k = sqrt(alpha_bar[k]) * actions + sqrt(1 - alpha_bar[k]) * epsilon
loss = MSE(network(x_k, k / (N - 1), observations), epsilon)
```

The cosine schedule is stored as a module buffer, included in `state_dict`, and
moves with the module. DDIM runs from the last noisy step to clean actions.
When clipping predicted clean actions, the noise direction is recomputed to
remain consistent with the clipped prediction. No stochastic DDPM sampler is
included.

Flow uses `t ~ Uniform(0,1)` and Gaussian `z`:

```text
x_t = (1 - t) * z + t * actions
velocity_target = actions - z
loss = MSE(network(x_t, t, observations), velocity_target)
```

At inference, solve `dx/dt = network(x,t,observations)` from `x(0)=z` to `t=1`.
Only the final flow sample is clipped. The implementation is vanilla conditional
flow matching, without optimal-transport matching, consistency distillation,
guidance, or a vision encoder.

For checkpoints, save the constructor configuration, `policy.state_dict()`, and
normalization statistics. The model does not own the optimizer, EMA, data loader,
environment, simulator, training loop, or evaluation pipeline. An experiment may
add these when training on real demonstrations.

## Verification

```sh
conda activate maniskill
python -m unittest discover -s models/policy/action_policy/tests -v
```

Tests verify the noising/interpolation targets, oracle DDIM reconstruction,
ODE accuracy against `dx/dt=x`, masking, invalid shapes, arbitrary horizons,
checkpoint round trips, reproducible sampling, gradients, and CUDA execution
when available. These are numerical/model checks, not evidence of learned robot
task success. No task training or rendering is run by the test suite.
