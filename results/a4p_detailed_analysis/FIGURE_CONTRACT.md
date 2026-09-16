# Diagnostic figure contract

- Core conclusion: numerical selectivity ratios depend on normalization and sampling; the present selected-path data do not uniquely determine a transferable alpha.
- Archetype: quantitative grid, with normalization as the leading panel.
- Backend: Python, matching the existing analysis workflow and saved preference.
- Output: 183 mm × 155 mm; editable SVG and 300 dpi PNG preview. This is a diagnostic research figure, not a claim of journal submission readiness.
- Panel a: compare contact/reference weight ratios with finite-domain normalized contact ratios; paired bootstrap intervals are conditional on the sampled histograms.
- Panel b: four successive 100 ps production blocks show time dependence.
- Panel c: move the reference interval without changing the contact interval or trajectories.
- Panel d: show the distant tail of the selected q1 cross pathways, explaining why finite-domain normalization is also incomplete.
- Statistics: one trajectory per window, 25 windows per pathway; 100 synthetic trajectory bootstraps per pathway, with independent pairing for ratio diagnostics. These intervals do not include between-start or hidden-basin uncertainty. No significance stars or p-values.
- Source data: local audit JSON and corresponding WHAM histograms and profiles; axes and units are unchanged by styling.
- Risks: selected pairs are not entire reaction classes; finite-domain probabilities are not complete-ensemble probabilities; A4 arm-number filtering is not a graph-defined neighbour rule.
- Integrity: no data smoothing, no omitted time blocks, no radial Jacobian correction added to the probability-density PMF, no image-generation model.
