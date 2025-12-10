# scripts/run_gen1_builders.py

import numpy as np
from pathlib import Path
from material_niches.env import GridConfig
from material_niches.simulate import run_generation

OUTDIR = Path("results") / "gen1_builders"
OUTDIR.mkdir(parents=True, exist_ok=True)

config = GridConfig(
    width=10,
    height=5,
    start_positions=[(4, 0)],  # (y, x)
    goal_positions=[(2, 9)],
)

env, trajectories = run_generation(
    config,
    builder_generation=True,
    n_episodes=50,
    T=40,
    seed=42,
)

# Save logs
np.savez(
    OUTDIR / "builder_runs.npz",
    tile_map=env.tile_map,
    trajectories=trajectories,
)
print(f"Saved builder generation logs to {OUTDIR}")
