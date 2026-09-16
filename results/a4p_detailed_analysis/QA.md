# Diagnostic QA

- Independently rebuilt all eleven full-25-window profiles with source TPR and pullx SHA-256 provenance, without trusting the earlier command-only cache.
- Confirmed 275 complete production windows. Recovery trajectories and their k=2000 TPR files are retained separately in the downloaded archives.
- Confirmed the A4 central-carbon split using the source heavy-atom graph; no cyclic scaffold order was inferred from numeric labels.
- Re-read all six states × three unbiased pilot replicas with atom-map bond-length and orthorhombic-PBC checks.
- Used equal contact/reference intervals in all quantitative comparisons. Finite-domain probability ratios are explicitly conditional, not complete-ensemble probabilities.
- No radial Jacobian was applied to a WHAM distance-probability PMF.
- Plotted every time block, with no smoothing. Bootstrap intervals describe synthetic within-histogram sampling only, not independent MD replicates or hidden-basin uncertainty.
- Python exclusively generated the SVG and PNG. The PNG was visually inspected: all four panels, units, log axes, time blocks, legends and interval definitions are readable, with no clipped data.
- Original molecular structures and simulation archives were not changed; no paid jobs were submitted in this audit.
