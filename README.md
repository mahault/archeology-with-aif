# archeology-with-aif
Archeology with AIF

## Recent Updates

### Navigation Fix (Dec 2024)

Fixed several issues that were preventing the active inference agent from reliably reaching goals:

1. **Deterministic Position Observations**: Changed position A-matrix from uncertainty-based (which penalized exploration by diluting C preferences) to deterministic (~98% certainty). Epistemic value now comes from tile uncertainty instead.

2. **Reduced Comfort Preferences**: Lowered C_land (0.5→0.1) and C_arousal (2.0→0.5) to prevent "comfort trap" where agent preferred staying on marked tiles over moving toward goals.

3. **Immediate Waypoint Model Updates**: Model now rebuilds immediately when waypoints are reached, not just every N steps. This prevents stale C_pos preferences from targeting old waypoints.

4. **Initial Waypoint Skip**: Added check to advance past waypoint 0 if agent starts there, so initial model targets the next waypoint.

5. **Smoothed D Prior**: Changed D_loc from point mass (100% at start) to smoothed prior (95% start + uniform baseline). Point mass priors can't be updated by observations since 0 × likelihood = 0.

Result: Agent now achieves **100% goal rate** with deterministic action selection.

---

## Roadmap
Phase 0 — Boilerplate / environment baseline

Set up repo

Implement a plain gridworld environment + simple pymdp agent following the official tutorial. 
pymdp-rtd.readthedocs.io
+1

Deliverables:

env.py basic GridWorld (no niche construction yet)

generative_model.py that builds A,B,C,D for a simple navigation task

run_simple_gridworld.py script and a sanity-check notebook

Phase 1 — Single generation with MODIFY / niche construction

Add tile condition factor and MODIFY action.

Implement Wild vs Marked A-matrix construction; add arousal modality if desired.

Implement environment logic: MODIFY flips the tile at current location from Wild→Marked for future time steps / episodes.

Deliverables:

run_gen1_builders.py that:

runs many episodes

logs locations modified

saves path density maps to results/trajectories and e.g. .npz logs.

Phase 2 — Multi-generation & path inheritance

Use Gen 1 outputs to seed Gen 2 environment.

Implement Gen 2 agents (no MODIFY; different D/C for priors).

Show difference in performance and EFE with vs without path inheritance.

Deliverables:

run_gen2_inheritors.py

Notebook showing:

average steps to goal, free-energy curves

path heatmaps for both generations

Phase 3 — Inference from absence (taphonomy)

Implement condition that removes a segment of Marked path in environment.

Keep agent priors unchanged.

Log trial-by-trial free energy, arousal, and location for those episodes.

Deliverables:

run_inference_from_absence.py

Notebook showing:

FE / arousal spike at “missing monument”

comparison between intact vs damaged path

Phase 4 — Salience / monument characteristics (Arturo’s suggestion)

Implement action restrictions or cost in certain tiles (e.g., “rough terrain”, “blocked corridors”).

Encode preferences to move toward next visible marker via C and possibly extra state factors.

Systematically vary:

salience (how deterministic A is for markers)

spatial configuration of obstacles / restricted tiles

Deliverables:

run_salience_sweep.py scanning over salience & constraint parameters

Notebook relating navigation success to “monument salience” and constraints