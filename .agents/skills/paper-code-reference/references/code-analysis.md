# Code analysis

Read this reference for source-grounded repository documentation or paper-to-code comparison. Trace the implementation that is actually present at the documented revision.

## Establish the source revision

Inspect any existing checkout and its changes first. Preserve the current pin unless the user authorizes an update. When obtaining a requested repository, follow the parent project's acquisition convention; add a submodule when requested or established rather than modifying unrelated repository structure.

Look for a paper-linked release/tag/commit. Use repository history, file locations, entry points and model/config changes to evaluate candidates. A repository URL is evidence of the requested source, not proof of author identity, completeness, or reproducibility. Record provenance accurately. If competing revisions materially change the research object and the user has not chosen one, present evidence and alternatives before depending on a choice.

Record the repository URL, full commit SHA, and any pre-existing modifications affecting the analysis. Check submodule declaration, gitlink and checkout consistency when applicable. Do not mutate source files just to make the checkout conform to the parent project's style or documentation rules.

## Follow entry points into behavior

Start at real command-line or API entry points. Trace what they call rather than inferring behavior from filenames, a README, or the paper alone. Cover the following where present; explicitly identify missing stages instead of inventing them:

| Stage | Questions to resolve |
| --- | --- |
| Data preparation | What source data and metadata are required? How are tasks/episodes split, sampled, aligned and windowed? What are boundary rules? |
| Preprocessing | What observations are used, in which order? How do resize/augmentation, normalization, tokenization, padding and masks work? |
| Model construction | What actual backbone/core networks, projectors, heads, retrieval modules or caches are connected? Which weights are frozen or trained? |
| Training | What targets, losses, reductions, masks, sampling schedules and optimizer/save behavior are implemented? Which artifacts are required later? |
| Inference | What is loaded, retrieved or recomputed? What are initialization, iteration and decoding behavior? Are weights updated? |
| Execution/evaluation | How are actions scaled and mapped to a controller/API? Which predictions are executed? How do feedback, stopping, success tests and recording work? |

Document the important tensor/data contracts: shape, dtype, temporal offsets, field order, units, normalization, masks, action meaning and embodiment/controller assumptions. Mark checkpoint-configured dimensions instead of presenting defaults as universal constants.

Trace artifacts through producer and consumer boundaries: raw/processed data, normalization statistics, retrieval memory, checkpoints, configuration, metadata, videos and logs. Explain which artifacts are external to the model checkpoint, which are regenerated, and which paths/identities must match. Distinguish training buffers, held-out demonstration buffers and evaluation outcomes.

For important findings, point to stable source paths and symbols; line numbers or immutable upstream links may supplement them. A file tree alone is not a code guide.

## Explain architecture and operation

Begin with a module-and-arrow architecture view that shows end-to-end data flow. Separate offline preparation/training from online inference. For robotic policies, include observation/action interfaces, preprocessing, the actual core network, action decoding/controller and environment feedback. For other domains, show their real external input/output and interactions instead of inventing a robot loop.

Add UML where it clarifies behavior:

- Class diagrams for important inheritance, composition and dependencies.
- Sequence diagrams for ordered calls, data exchange, loss branches and inference/control loops.
- Activity/state diagrams for meaningful decisions or artifact/runtime lifecycle.

Choose useful views, not a mandatory diagram count or class hierarchy. Include key dimensions and semantics where they explain the connections; avoid unreadable diagrams packed with every file or layer. Reflect actual network choices such as MLP, U-Net, Transformer, DiT or flow-matching head.

Keep editable sources beside the documentation, under version control. Embed Mermaid when the intended reader supports it; otherwise provide a readable rendered view with its editable source. If external diagram sources and embedded copies coexist, compare them. Describe the actual validation level; source consistency does not prove the diagram parsed or rendered.

## Separate claims and limitations

Distinguish three evidence levels throughout the code guide:

1. What the paper states.
2. What the pinned source implements or statically implies.
3. What was actually executed and observed during this task.

Explain material mismatches such as missing inputs, different network/configuration, unsupported parameter ranges, omitted masks, different cache schema, stale imports, absent entry points, default checkpoint mismatches or changed action semantics. Record uncertainty rather than asserting a runtime defect without evidence. Link each important discrepancy to its source.

Documenting an upstream defect is not authorization to repair it. Do not install packages, download weights/data or run an experiment merely to make documentation look complete. If execution is requested separately, apply project execution rules and report its results separately from the paper copy.

## Overview and operation notes

Update the destination README or equivalent with document navigation, source revision and operation requirements. Inspect actual CLI definitions before showing commands: entry point, required inputs, output locations and result inspection must match the source. Prefer the project's module execution convention when it is supported; do not inject import paths or invent flags to compensate for missing packaging.

State when an example is only a checked description of an upstream interface and cannot yet run because of missing configuration, packaging, dependencies, platform support or artifacts. If paths are hard-coded, expose that limitation rather than claiming caller control. Keep commands and environment details in the owning operation documentation; do not copy them into permanent project-wide principles.
