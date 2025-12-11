# scripts/run_gen1_builders.py

from pathlib import Path
import numpy as np

from material_niches.env import GridConfig, NicheGridEnv
from material_niches.simulate import run_generation

def main():
    outdir = Path("results") / "gen1_builders"
    outdir.mkdir(parents=True, exist_ok=True)

    config = GridConfig(
        width=10,
        height=5,
        start_positions=[(4, 0)],  # (y, x)
        goal_positions=[(2, 9)],
    )

    env, trajectories = run_generation(
        config,
        builder_generation=True,
        n_episodes=20,
        T=30,
        seed=42,
    )

    arr_loc = np.stack([traj["loc"] for traj in trajectories], axis=0)
    arr_tile = np.stack([traj["tile"] for traj in trajectories], axis=0)
    arr_act = np.stack([traj["action"] for traj in trajectories], axis=0)
    arr_fe = np.stack([traj["free_energy"] for traj in trajectories], axis=0)

    np.savez(
        outdir / "builder_runs.npz",
        tile_map=env.tile_map,
        loc=arr_loc,
        tile=arr_tile,
        action=arr_act,
        free_energy=arr_fe,
    )

    print(f"Saved builder generation logs to {outdir / 'builder_runs.npz'}")

if __name__ == "__main__":
    main()
