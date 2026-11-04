# Paper transcription

Read this reference when a complete paper copy is requested. The result must preserve the supplied work, not replace it with a summary or a new explanation.

## Inventory before extraction

Identify the actual document/version, title, authors/affiliations, source identifier, page count and file hash. Inspect every page, including unnumbered pages and supplementary material that is part of the supplied source. Build a page-to-content inventory of sections, figures, tables, numbered equations, algorithms, footnotes, bibliography and appendices. Include meaningful unnumbered visuals and text.

Choose tools from what is installed. Prefer native text extraction for searchable PDFs, with rendered-page inspection to establish reading order and verify visuals. For scanned pages or damaged extraction, use available OCR or direct visual transcription and check it against the page. If a required source is missing or a region remains unreadable, identify the exact gap and seek the necessary input; do not fill it from memory.

A format-specific PDF skill can help when available, but the final Markdown must not require that skill to be read.

## Transcribe logically

- Recover logical column order, paragraphs, section hierarchy and sentences continued across pages. Do not interleave neighboring columns or place figure labels inside body sentences.
- Retain the source language unless translation is requested. Preserve title/author information, acknowledgements, impact statements, footnotes, bibliography and appendices.
- Repair extraction artifacts such as discretionary line-break hyphens, ligatures, missing spaces and superscript order. Preserve lexical hyphens and meaningful punctuation.
- Express equations in readable math markup with original identifiers, operators, superscripts/subscripts, indices, ranges and numbering. Keep explanatory definitions attached. Transcribe algorithms/pseudocode as structured text or code-like blocks, including branch and loop semantics.
- Preserve citations and join wrapped bibliography URLs carefully. Do not replace bibliographic entries with links alone.
- Preserve apparent source mistakes or contradictory units/values. Add clearly labeled editorial notes only when useful, rather than silently correcting the paper.
- Repeated running headers/page numbers and clipped off-page layout residue may be omitted if they carry no substantive content. State that omission policy. Never classify an unexplained content gap as layout residue.

Use stable figure/table/equation anchors where they improve navigation. Keep editorial metadata distinct from source text. A full transcription should remain a faithful copy even when the code guide later disputes implementation details.

## Preserve tables as data

Use Markdown tables where possible; use necessary HTML for merged headers, multiline cells or other structure that Markdown cannot express clearly. Keep every row/column, hierarchy, value, sign, decimal precision, unit, sample size, note and meaningful emphasis. Do not round values, convert fractions to percentages, drop wide columns or use a screenshot as the only table representation.

When restructuring a complex table, preserve grouping explicitly in the header labels or notes. Compare numeric and textual cells with the actual page. Automated cell/shape comparison is useful, but visual inspection is still needed for merged headers, superscripts, emphasis and footnotes.

## Preserve figures as visuals

Preserve each complete visual composition, including all panels, arrows, legends, axes, labels and in-image annotations. Native extracted assets are useful only if they retain that complete composition; raster page-region crops may be needed for mixed raster/vector figures. Do not use generated images or newly drawn approximations as substitutes.

Choose resolution based on label readability at useful viewing size, rather than a universal DPI. Avoid clipped labels, adjacent body text, neighboring figures, and unexplained duplicate assets. Visually inspect every delivered figure, including chart labels and plot boundaries. A contact sheet helps initial coverage; inspect detailed crops when labels or boundaries are uncertain.

Store deliverable images beside the paper copy in its document resources and reference every delivered visual. Keep figure numbering, full captions, and notes in searchable Markdown, not only baked into the image. Temporary full-page previews are QA resources unless the user asks to retain them.

## Provenance and coverage

Reuse an existing manifest convention where available. Otherwise keep a small source/asset manifest with:

- Source identity, original relative filename, document version, hash algorithm/value and total source pages.
- Each figure's original identifier, source page, output relative path and asset hash.
- For crops: page coordinate system, origin, units, crop rectangle and rendering scale/resolution. These must be sufficient to locate the source region without guessing.
- A source-to-output coverage ledger for page/section/table/equation/figure/footnote/bibliography/appendix content. A compact range/list mapping is enough; it need not contain a duplicate of the paper text.
- Explicit unresolved regions or deliberate non-content omissions, if any.

Do not encode one paper's expected counts into reusable project rules. Counts and checks belong to that document's provenance or verification record. If original embedded assets are specifically requested, preserve those as well as any complete composite representations and explain their relationship.
