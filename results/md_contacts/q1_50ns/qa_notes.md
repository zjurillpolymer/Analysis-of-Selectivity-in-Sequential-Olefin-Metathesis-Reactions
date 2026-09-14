# QA notes

- Three q1 XTC files were read by GROMACS to 50,000 ps with 100,001 frames at 0.5 ps spacing.
- All three extension markers report `status=complete`, 50 ns total duration, and checkpoint continuation without velocity reinitialisation.
- Distance extraction read 300,003/300,003 frames and 45 candidate-pair columns per replica.
- The structural quality gate found no missing frames, duplicate times, non-finite distances, or non-positive distances.
- All 45 q1 pairs pass the prespecified split-Rhat/ESS convergence rule; all 12 material pairs pass.
- Figure bars use independent replicas as the uncertainty unit; frame-wise p-values are intentionally not reported.
- q0 and q1 arm-pair sums have different active-pair counts when A or D is involved. The source table therefore reports both total opportunity and mean per active candidate.
- SVG/PDF/TIFF/PNG were generated from the same Python script and source tables; no manual plot editing was applied.
- Visual QA was performed on the final PNG at native resolution after moving occupancy labels away from replica points.
- The SVG contains editable text nodes; the PDF embeds subset TrueType fonts and is unencrypted.
- The TIFF export is 4317 × 3819 pixels at 600 dpi; the PNG preview is 2158 × 1909 pixels at 300 dpi.
