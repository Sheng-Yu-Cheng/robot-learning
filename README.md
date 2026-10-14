# Robot Learning

Robot Learning is a personal, executable knowledge base for learning, implementing, and experimenting with ideas in artificial intelligence.

The repository is organized around the kinds of things I may want to learn:

- **Methods** — algorithms and concrete procedures
- **Models** — model architectures and representational structures
- **Paradigms** — problem settings and broader ways of formulating learning
- **Datasets** — reusable data resources
- **Benchmarks** — standard tasks, task collections, and evaluation settings
- **Experiments** — concrete things I actually run

The goal is **not** to build a universal AI framework.

Different ideas may use completely different data formats, training loops, environments, simulators, hardware, and evaluation procedures. Robot Learning should preserve those differences instead of hiding them behind abstractions that do not naturally fit.

---

## Repository Structure

```text
robot-learning/
├── methods/
├── models/
├── paradigms/
├── datasets/
├── benchmarks/
├── experiments/
├── tests/
├── README.md
├── AGENTS.md
├── setup.py
└── .gitignore
```

The top-level directories are the knowledge structure of the repository.

There is intentionally no `src/robot_learning/` wrapper.

---

## Methods

`methods/` contains concrete algorithms or methods.

Examples:

```text
methods/
├── value_iteration/
├── q_learning/
├── sarsa/
├── dqn/
├── ppo/
├── sac/
├── ddim/
├── decision_transformer/
├── pdt/
└── ...
```

A method answers:

> How is this actually done?

Examples include:

- Value Iteration
- Q-Learning
- SARSA
- PPO
- SAC
- DDIM
- CEM
- MCTS
- Decision Transformer
- PDT
- DPO

A method entry owns the reusable logic needed to understand and implement that method.

Its internal structure is intentionally flexible.

A small method may look like:

```text
methods/q_learning/
├── README.md
├── AGENTS.md
└── q_learning.py
```

A larger method may look like:

```text
methods/pdt/
├── README.md
├── AGENTS.md
├── pdt.py
├── model.py
├── losses.py
├── dataset.py
├── config.yaml
└── tests/
```

Different methods do **not** need to expose the same API.

There is no required `BaseAlgorithm`, `Agent`, `Trainer`, `Runner`, `act()`, `update()`, or `train()` interface.

---

## Models

`models/` contains models and architectures that are worth studying independently.

Examples:

```text
models/
├── mlp/
├── cnn/
├── resnet/
├── transformer/
├── unet/
├── diffusion_model/
├── world_model/
├── world_action_model/
└── ...
```

A model answers:

> What computational structure represents the function being learned?

Not every neural network used inside a method needs to become a separate model entry.

For example, a small actor-critic network used only by one PPO implementation can remain inside:

```text
methods/ppo/
```

A model should become its own entry when the architecture itself is worth understanding, implementing, comparing, or reusing.

---

## Paradigms

`paradigms/` contains broader problem settings and ways of thinking about learning problems.

Examples:

```text
paradigms/
├── reinforcement_learning/
├── offline_rl/
├── model_based_rl/
├── meta_rl/
├── in_context_learning/
├── in_context_rl/
├── world_models/
└── ...
```

A paradigm answers:

> What kind of problem are we solving, and under what assumptions?

A paradigm entry may explain:

- the problem setting
- assumptions
- objectives
- important terminology
- standard pipelines
- major challenges
- taxonomy of approaches
- representative methods
- relevant models
- common datasets
- common benchmarks

Paradigms are primarily conceptual and may contain little or no executable code.

For example:

```text
paradigms/offline_rl/
└── README.md
```

may connect concepts such as distribution shift and extrapolation error to concrete entries such as CQL, IQL, Decision Transformer, PDT, D4RL, and offline-control benchmarks.

---

## Datasets

`datasets/` contains data resources that are worth understanding and reusing.

Examples:

```text
datasets/
├── d4rl/
├── minari/
├── robomimic/
├── libero/
├── open_x_embodiment/
└── ...
```

A dataset entry may document:

- where the data comes from
- how it was collected
- data schema
- observations, actions, rewards, labels, or language fields
- train / validation / test splits
- preprocessing
- download or loading code
- known issues
- example usage

Large datasets should normally stay outside Git.

The repository should store the knowledge and code needed to obtain, inspect, and use them.

A dataset and a benchmark with the same project name may both exist when they represent different things.

For example:

```text
datasets/libero/
benchmarks/libero/
```

is valid: one describes the data resource, the other describes the evaluation setting.

---

## Benchmarks

`benchmarks/` contains standard problems and evaluation settings.

Examples:

```text
benchmarks/
├── cartpole/
├── cliff_walking/
├── classic_control/
├── atari_57/
├── pointmaze/
├── push_t/
├── d4rl_locomotion/
├── metaworld/
└── libero/
```

A benchmark answers:

> What do we test on, and how is performance interpreted?

A benchmark entry may represent:

- a single task, such as CartPole
- a collection of tasks, such as Classic Control or Atari-57
- a benchmark built around a dataset
- a task suite
- an evaluation protocol
- benchmark-specific metrics or scoring rules

These are all benchmark entries. Robot Learning does **not** force separate `tasks/`, `suites/`, and `protocols/` directory hierarchies.

Composition belongs in the benchmark documentation itself.

For example:

```text
benchmarks/cartpole/
```

may describe the single CartPole task, while:

```text
benchmarks/classic_control/
```

may describe a collection containing CartPole, Acrobot, MountainCar, and other tasks.

The benchmark structure does not imply a shared runtime API.

There is no universal `BaseEnv`.

A benchmark may use Gymnasium, MuJoCo, MetaWorld, ManiSkill, a custom simulator, a dataset, a real robot, or anything else appropriate to the problem.

---

## Experiments

`experiments/` contains concrete things that I actually run.

Examples:

```text
experiments/
├── q_learning_cliff_walking/
├── sarsa_cliff_walking/
├── ppo_cartpole/
├── dqn_cartpole/
├── pdt_pointmaze/
├── diffusion_policy_push_t/
└── wam_pointmaze/
```

An experiment answers:

> What did I actually run?

An experiment is where entries from the rest of Robot Learning are combined.

For example:

```text
methods/ppo/
        \
         \
          > experiments/ppo_cartpole/
         /
        /
benchmarks/cartpole/
```

A typical experiment may contain:

```text
experiments/ppo_cartpole/
├── README.md
├── AGENTS.md
├── train.py
├── evaluate.py
├── config.yaml
└── outputs/
```

The experiment owns its execution context.

That may include:

- environment creation
- training loop
- dataset loading
- simulator setup
- robot connection
- hyperparameters
- preprocessing
- checkpointing
- plotting
- evaluation

Different experiments do **not** need to share the same environment interface or training loop.

For example, PPO on CartPole may simply use:

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
```

while a robotics experiment may directly connect to robot hardware, cameras, and its own control loop.

That difference is intentional.

---

## Example: PPO on CartPole

A PPO implementation may live in:

```text
methods/ppo/
├── README.md
├── AGENTS.md
├── ppo.py
├── network.py
└── buffer.py
```

CartPole as a benchmark may live in:

```text
benchmarks/cartpole/
└── README.md
```

The concrete experiment lives in:

```text
experiments/ppo_cartpole/
├── README.md
├── AGENTS.md
├── train.py
├── evaluate.py
└── config.yaml
```

`methods/ppo/` should contain reusable PPO logic.

`benchmarks/cartpole/` should describe the benchmark.

`experiments/ppo_cartpole/` owns the actual CartPole environment, hyperparameters, training loop, checkpointing, and evaluation code used in that experiment.

There is no need for PPO and CartPole to communicate through a repository-wide base class.

---

## Classification Examples

Some examples of where concepts belong:

| Concept | Location |
| --- | --- |
| Q-Learning | `methods/q_learning/` |
| PPO | `methods/ppo/` |
| DDIM | `methods/ddim/` |
| Decision Transformer | `methods/decision_transformer/` |
| PDT | `methods/pdt/` |
| Transformer | `models/transformer/` |
| U-Net | `models/unet/` |
| World Action Model | `models/world_action_model/` |
| Reinforcement Learning | `paradigms/reinforcement_learning/` |
| Offline RL | `paradigms/offline_rl/` |
| Meta-RL | `paradigms/meta_rl/` |
| In-Context RL | `paradigms/in_context_rl/` |
| D4RL data | `datasets/d4rl/` |
| Open X-Embodiment | `datasets/open_x_embodiment/` |
| CartPole | `benchmarks/cartpole/` |
| Classic Control | `benchmarks/classic_control/` |
| Atari-57 | `benchmarks/atari_57/` |
| PPO on CartPole | `experiments/ppo_cartpole/` |

When a concept could fit more than one category, choose the canonical home based on what the entry is primarily trying to teach or implement.

Cross-category relationships can be documented in the entry's README instead of forcing the filesystem to encode every taxonomy.

---

## Design Principles

### 1. Robot Learning is an executable knowledge base, not a universal framework

The repository exists to make AI ideas understandable, implementable, testable, and usable.

It does not exist to make every AI idea look the same.

---

### 2. No universal environment abstraction

There is no `BaseEnv`.

Each experiment or benchmark may use whatever environment, simulator, dataset, hardware, or API is appropriate.

Environment-specific code can live directly inside the experiment or benchmark that needs it.

---

### 3. No universal method abstraction

There is no required `BaseAgent`, `BaseAlgorithm`, `Trainer`, `Runner`, or shared lifecycle.

Q-Learning, PPO, DDIM, Decision Transformer, MCTS, and World Model training are allowed to look fundamentally different.

---

### 4. No universal model abstraction

A Transformer, tabular Q-function, World Action Model, diffusion network, and hand-written planner do not need to inherit from a shared repository abstraction.

Use the native abstraction of the relevant framework when useful, such as `torch.nn.Module`.

---

### 5. Experiments own execution

Concrete execution belongs to the experiment.

Reusable method or model logic belongs to the corresponding knowledge entry.

This keeps reusable ideas independent from one particular environment or benchmark without requiring a global runtime framework.

---

### 6. Do not create abstractions before they are needed

Prefer:

```text
implement
→ see real repetition
→ extract shared code
```

over:

```text
predict future abstractions
→ build infrastructure
→ force future work into it
```

Duplication is acceptable until a useful common abstraction becomes obvious.

---

### 7. Entries do not need identical internal layouts

A simple Q-Learning entry and a large research method may have completely different directory structures.

That is expected.

Only create files that the entry actually needs.

---

### 8. Keep explanation and implementation close together

Whenever practical, an entry should contain:

- a `README.md` explaining the idea
- executable code implementing or demonstrating it

Robot Learning should be neither a notes-only repository nor a code-only collection.

---

### 9. The filesystem stores canonical knowledge entries, not every relationship

Do not build deep directory taxonomies just to express relationships such as:

```text
task → suite
algorithm → paradigm
dataset → benchmark
model → method
```

Document those relationships in README files and experiment organization.

For example, CartPole and Classic Control can both be first-class entries under `benchmarks/`, even though one may be contained by the other.

---

## Initial Focus

The initial focus of Robot Learning is reinforcement learning.

Early method implementations may include:

```text
Value Iteration
Q-Learning
SARSA
Expected SARSA
Double Q-Learning
Dyna-Q
REINFORCE
DQN
PPO
SAC
```

Early benchmarks may include:

```text
Cliff Walking
CartPole
Acrobot
MountainCar
Classic Control
MuJoCo
```

The repository should later be able to grow naturally into topics such as:

```text
Transformers
Diffusion Models
DDIM
Flow Matching
World Models
World Action Models
Offline RL
Decision Transformers
PDT
Meta-RL
In-Context RL
VLA
Robot Learning
```

without redesigning the repository around a universal framework.

---

## Philosophy

Robot Learning is a place to accumulate AI knowledge through implementation and experimentation.

The repository should make different ideas:

**understandable, executable, reusable, and comparable — while preserving what makes them different.**

## Existing Implementations

The implementations brought over from AI-Lib currently live at:

```text
benchmarks/gymnasium/continuous_mountain_car.py
methods/reinforcement_learning/actor_critic/network.py
methods/reinforcement_learning/advantage_estimator/advantage_estimator.py
experiments/reinforcement_learning/actor_critic_mountain_car/main.py
experiments/reinforcement_learning/actor_critic_mountain_car/train.sh
```

These existing paths are preserved so their imports continue to work. The other examples above describe how the knowledge base can grow; they are not implemented entries.

The MountainCar experiment uses PyTorch and Gymnasium. After preparing your environment and installing this project with `python -m pip install -e .`, run it from the repository root:

```sh
python experiments/reinforcement_learning/actor_critic_mountain_car/main.py
```

The original `train.sh` is also preserved. It manages its own conda environment named `ai_lib-actor_critic_mountain_car`; review its environment and installation commands before using it. Environment setup is managed by the user.

## GPU Base Environment

`gpu-base` was verified on 2026-10-14: Python 3.11.17, PyTorch 2.11.0+cu128,
CUDA runtime 12.8, NVIDIA GeForce RTX 3060 Ti (8 GiB). GPU matrix multiplication,
CPU result comparison, and backward propagation passed. Dependency checks passed.

`envs/gpu-base.yml` combines the conda and pip requirements. Python is constrained
to 3.11, and the Python package versions are pinned. Platform-specific system
libraries and build identifiers are left to conda to resolve. The CUDA wheel
requires a supported Windows/Linux machine with an NVIDIA GPU and driver;
macOS requires a different PyTorch package specification.

Create an independent experiment environment from the YAML:

```sh
conda env create -n robot-learning-exp -f envs/gpu-base.yml
conda activate robot-learning-exp
python -m pip install -e .
```

Alternatively, clone the existing local baseline:

```sh
conda create -n robot-learning-exp --clone gpu-base
```

Replace `robot-learning-exp` with the experiment's environment name. Keep the
baseline unchanged and add experiment dependencies to the new environment.
NumPy and Gymnasium are not included; the existing MountainCar experiment needs
them. This YAML specifies compatible dependencies rather than identical package
builds or wheel hashes, so a rebuilt environment still needs validation.
The conversion does not modify the existing `gpu-base` environment.
