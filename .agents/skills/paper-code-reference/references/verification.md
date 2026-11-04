# Verification and maintenance

Read this before delivery, and when refreshing an existing reference. Choose checks that match the actual changes; do not turn a documentation task into an experiment.

## Paper completeness

Reconcile the source inventory and page-to-output ledger with the Markdown and assets. Check every source page for substantive content not represented in the output. A section heading list or matching figure count alone does not prove completeness.

- Compare body paragraphs, equations, algorithms, footnotes, bibliography and appendices in logical order. Check cross-page continuations, repaired hyphens and mathematical superscripts/subscripts against the page.
- Check table structure, headers, all cells, signs, precision, units and notes. Automated extraction comparisons can help but do not establish merged-header semantics.
- Visually inspect every delivered figure for full panels, labels, chart boundaries, readability and matching caption. Inspect detailed crops where a contact sheet is insufficient.
- Verify original identifiers/anchors, internal references and local links. Every delivered figure should have a reference; references must resolve.
- Recompute source/asset hashes and check page/crop provenance. Record unresolved source regions explicitly. A correct hash proves identity, not accurate transcription.

Mark each expected item as preserved, deliberately omitted non-content layout, or unresolved. Do not deliver a partial extraction as a full paper copy. A source disagreement should remain visible as an editorial note, not a silent alteration.

## Code evidence

Confirm that documented source paths/symbols exist at the recorded SHA. Cross-check entry-point defaults, required artifacts, dimensions, transforms, loss/inference branches and controller interfaces against calls and configuration. Check whether a value is fixed, a script default, or checkpoint-configured.

For submodules, verify URL/declaration, gitlink and checkout identity. Preserve source changes that predate the task and distinguish them from the pinned upstream state. A clean checkout does not prove the source matches the paper or runs successfully.

Check diagrams against the described implementation and their editable sources. When a parser/renderer is available, validate syntax and inspect rendered output. If it is unavailable, state that limitation. Do not install a renderer without authorization or claim rendering from a text comparison.

Keep important paper/source/observed-result differences explicit. Static Python syntax/import inspection is optional when it addresses a real concern; parsing is not a full import test and imports may have side effects. Do not import model or simulator packages just for documentation validation.

## Portable artifact checks

Use relative paths inside the delivered reference. Verify links without executing commands found in linked source content. Keep machine-specific absolute paths, credentials, checkpoints, datasets and caches out of the skill package. Example paths in operation notes must clearly be caller-supplied inputs, not claims about the reader's machine.

Create no placeholder assets, unused resource directories or duplicate paper copies. Avoid generic automation frameworks. Add reusable checking scripts only after concrete repeated work or a fragile invariant justifies them; verify meaningful behavior if scripts are added.

## Reporting and cleanup

Report checks with their scope:

| Evidence | What it establishes |
| --- | --- |
| Source/page coverage and visual comparison | Which paper content and visual assets were reviewed/preserved. |
| Link/hash/source consistency | Artifact identity and reference integrity. |
| Parsing or static inspection | Limited syntactic/source properties. |
| Rendered diagram/document review | Presentation of the exact rendered artifact. |
| Executed pipeline | Only the inputs, environment and path actually exercised. |
| Experiment reproduction | Measured outcomes from a documented run, not values copied from the paper. |

Keep a compact verification record in the destination documents or provenance. Do not store paper-specific counts, trials, parameters, errors or results in the reusable skill or project AGENTS.md.

Use temporary locations for extraction, previews and checker artifacts, track ownership, and remove only resources created for this task after verification. Retain the source PDF, final documents/images/diagram sources, user data and pre-existing upstream tests. Validate cleanup paths before recursive removal and follow the local shell rules. Do not provide links to deleted previews.

## Refreshing the reference

Update the source pin only when authorized. Re-trace changed entry points, interfaces, artifact schemas, core network choices and call connections. Revise code documentation, diagrams and recorded revision together. A network or I/O change requires an identifiable architecture update. Do not rewrite an unchanged paper transcription to make it agree with newer code.

For a paper-version update, compare source inventories before replacing assets; update provenance, captions, tables and references together. Preserve existing work or follow the user's explicit replacement/versioning choice. Treat an incomplete upstream release as a documented limitation until separately asked to integrate or repair it.
