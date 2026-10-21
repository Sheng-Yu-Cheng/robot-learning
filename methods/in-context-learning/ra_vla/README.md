# RA-VLA paper and code reference

This folder keeps a searchable full copy of the supplied paper and a source-grounded guide to its retrieval-augmented VLA implementation.

| Reference | Contents |
| --- | --- |
| [ravla.md](ravla.md) | All paper sections, acknowledgements, impact statement, references, appendices A–H, 5 Markdown tables, equations 1–10 and 11 figure references. |
| [images](images/paper-assets.json) | Complete figure crops at 200 DPI, with source PDF hash, page/crop coordinates and individual image hashes. |
| [ravla-code.md](ravla-code.md) | Code responsibilities, tensor contracts, retrieval/training/serving/control logic, UML diagrams and source limitations. |
| [diagrams](diagrams/01-system.mmd) | Editable Mermaid sources, also embedded in the code guide. |
| [RA-VLA](RA-VLA/README.md) | Unmodified third-party Git submodule. |
| [Original PDF](<RA-VLA Retrieval-Augmented VLA for Test-Time Adaptation.pdf>) | Supplied 18-page source, unchanged. |

The submodule is pinned to **`acb2a345c91af758f44643cedbc92dbe5d23c284`** (22 September 2026), selected to retain the paper’s N1.5 module organization. Later upstream revisions migrate the implementation to `gr00t_old/` without updating the RA-VLA scripts’ imports. Do not advance the pin as an incidental documentation update.

Read the Markdown directly or use a viewer supporting Mermaid and LaTeX math. Tables and body text are editable Markdown; figures remain visual assets with full captions. `paper-assets.json` maps each crop back to the source page. The diagrams describe this exact source revision and distinguish offline training from inference.

## Operating the upstream pipeline

**Runtime status:** documented, not executed. No environment was created, packages installed, weights fetched, training started or simulator run for this task. The upstream checkout is incomplete as an execution package. Resolve the blockers below in a separately authorized implementation/setup task before treating the commands as a runnable reproduction recipe.

- Stage 1 hard-codes an absent `examples.UR5e.custom_data_config:Ur5eDataConfig`. It needs a deliberate source/config repair for the intended task; no CLI data-config flag exists.
- This revision supplies no `pyproject.toml`, `setup.py`, dependency lock or complete environment specification. It needs a supported editable-package setup; do not add `sys.path` or `PYTHONPATH` workarounds.
- The local Eagle config and requested N1.5 checkpoint require a compatibility check. A supported CUDA/bf16 stack and model attention backend must be established independently; Windows or CPU support has not been demonstrated.
- Input datasets must be prepared in the expected LeRobot layout, with valid modality/episode/task/statistics metadata and two RGB streams. Pre-create each dataset’s `.cache` directory before video caching. Split complete episodes/tasks externally; the scripts do not enforce held-out splits or temporal-target masking.
- Use a separate experiment environment. Source imports require PyTorch, torchvision/video decoding, Transformers, Diffusers, Hugging Face Hub, NumPy, pandas/parquet support, Pydantic, dm-tree, einops, Pillow, tyro, tqdm, dtaidistance, ZeroMQ/MessagePack and the relevant model backend. Evaluation additionally needs LIBERO/robosuite/MuJoCo, OpenCV, imageio and an MP4 encoder. Exact compatible versions are not supplied upstream. This is an import-derived inventory, not an installation lock.

The following commands record the **actual dataclass CLI interfaces** at the pin. They are intentionally not advertised as verified executable before those blockers are repaired. Run modules only after the parent project and third-party packages have valid editable installations in the intended environment. All application data/output paths below are caller-selected absolute paths; replace the example roots. These shell snippets are for bash/zsh.

```bash
RA_FOLDER=/absolute/path/to/robot-learning/methods/in-context-learning/ra_vla
RA_RUN="$RA_FOLDER/outputs/libero_goal/paper_reference_run_001"
TRAIN_DATA=/absolute/path/to/prepared/training_tasks
HELDOUT_DATA=/absolute/path/to/prepared/heldout_goal_tasks
```

Keep raw downloads, processed demonstrations, retriever artifacts, training checkpoints and evaluation results separate. Use a new `RA_RUN` value for every batch; no example directory is created by this documentation task.

### 1. Train the retriever

After repairing/selecting the missing data config, train on **training tasks only**. The script uses frozen `nvidia/GR00T-N1.5-3B` embeddings and optimizes its small retriever. More than one dataset path can follow `--dataset-paths`.

```bash
python -m scripts.rag.1_train_retriever \
  --dataset-paths "$TRAIN_DATA" \
  --retriever-path "$RA_RUN/retriever" \
  --seed 0 --cuda 0
```

Inspect `messages.log`, saved retriever weights/config and `metadata.pt` in the retriever directory. This script has no validation-success evaluation or resume-state output.

### 2. Build the policy-training buffer

```bash
python -m scripts.rag.2_build_memory \
  --dataset-paths "$TRAIN_DATA" \
  --retriever-path "$RA_RUN/retriever" \
  --memory-name train_memory \
  --num-trajs-per-task 3 --frame-stride 4 \
  --seed 0 --cuda 0
```

The output is `$RA_RUN/retriever/train_memory.pt`. It uses the retriever’s saved normalization metadata. Each task needs enough trajectories and at least K cached segments. The memory schema is documented in [the code guide](ravla-code.md#stage-2-expert-memory).

### 3. Train the policy

```bash
python -m scripts.rag.3_train_ravla \
  --dataset-paths "$TRAIN_DATA" \
  --retriever-path "$RA_RUN/retriever" \
  --memory-name train_memory \
  --vla-path "$RA_RUN/policy_training" \
  --num-epochs 3 --k 1 \
  --seed 0 --cuda 0
```

Inspect `$RA_RUN/policy_training/messages.log` and saved `checkpoint_<epoch>` directories. The example requests three epochs; an explicit `checkpoint_3` is used below because the serving default `checkpoint_5` does not match the training default. Policy checkpoints need their companion metadata, retriever and memory. No formal result or experimental configuration is implied by these illustrative interface commands.

### 4. Build a separate held-out-task buffer

No retriever/policy weights are updated in this step. Only held-out demonstration segments are encoded using the already-trained retriever and its normalization.

```bash
python -m scripts.rag.2_build_memory \
  --dataset-paths "$HELDOUT_DATA" \
  --retriever-path "$RA_RUN/retriever" \
  --memory-name heldout_memory \
  --num-trajs-per-task 3 --frame-stride 4 \
  --seed 0 --cuda 0
```

The distinct `$RA_RUN/retriever/heldout_memory.pt` preserves the training buffer. Verify its task strings match evaluation instructions exactly. Buffer trajectory identities must be recorded externally for a reproducible experiment.

### 5. Serve RA-VLA

```bash
python -m scripts.rag.4_serve_ravla \
  --retriever-path "$RA_RUN/retriever" \
  --memory-name heldout_memory \
  --checkpoint-path "$RA_RUN/policy_training/checkpoint_3" \
  --denoising-steps 4 --k 1 \
  --host 127.0.0.1 --port 8071 \
  --seed 0 --cuda 0
```

This starts a ZeroMQ inference service and prints its listening address. It does not run evaluation or write policy rollouts. Both client and server must use port 8071 in this example.

### 6. Evaluate LIBERO offscreen

The real entry is `examples.Libero.eval.run_libero_eval`, not the missing path in the upstream root README. The example uses 50 trials/task to match the paper’s LIBERO trial count, without claiming that the rest of the published experiment has been reproduced.

```bash
mkdir -p "$RA_RUN/evaluation/videos"
cd "$RA_RUN/evaluation/videos" && \
python -m examples.Libero.eval.run_libero_eval \
  --task-suite-name libero_goal \
  --num-trials-per-task 50 \
  --port 8071 --headless
```

Use only a verified compatible headless/offscreen backend. `--headless` suppresses the OpenCV preview. The source writes MP4s to `./rollouts/<date>/` beneath the selected video working directory, and a fixed log to `/tmp/logs/libero_eval_libero_goal.log`. It does not accept `--output-dir` or `--log-dir`. These hard-coded paths prevent full compliance with the parent project’s explicit output-path interface and can overwrite logs; an authorized integration should expose them before formal evaluation. Do not invent unsupported flags.

Review the success counts and every episode’s side-by-side camera video. Validate non-empty video, frame count and duration before delivering evaluation results. The source predicts 16 actions but executes only the first, then queries again. Its 30-fps recording rate is separate from simulator/control timing.

## Maintaining this reference

Keep all local documentation outside the submodule. Changes to logic or I/O require corresponding code-guide and diagram updates. Regenerate paper crops only against an identified source PDF, update the provenance hashes, and retain all table/figure numbering. A paper-copy update does not authorize model execution or an upstream revision change.

Documentation checks cover local links, five table shapes/numbers, all eleven figure references/hashes, ten equation numbers, footnotes, full source-text coverage, Mermaid source consistency and the gitlink pin. Temporary extraction/render/check files are removed after verification. No execution-generated data is retained in this folder.
