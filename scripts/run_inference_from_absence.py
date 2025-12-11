# scripts/run_inference_from_absence.py

from pathlib import Path
import numpy as np

from material_niches.env import GridConfig, NicheGridEnv
from material_niches.generative_model import make_generative_model
from material_niches.agents import make_agent
from material_niches.simulate import run_episode

def main():
    gen1_file = Path("results/gen1_builders/builder_runs.npz")
    if not gen1_file.exists():
        raise FileNotFoundError("Run Gen1 first: results/gen1_builders/builder_runs.npz not found")

    data = np.load(gen1_file, allow_pickle=True)
    builder_tile_map = data["tile_map"].copy()

    outdir = Path("results") / "inference_from_absence"
    outdir.mkdir(parents=True, exist_ok=True)

    config = GridConfig(
        width=10,
        height=5,
        start_positions=[(4, 0)],
        goal_positions=[(2, 9)],
    )

    # base env with full inherited path
    env_full = NicheGridEnv(config, initial_tile_map=builder_tile_map)

    # taphonomy: remove a segment of the path around the middle of the corridor
    env_broken = env_full.clone_with_tilemap(builder_tile_map)
    cells_to_remove = [(2, x) for x in range(4, 7)]  # (y, x) in middle row
    env_broken.remove_path_segment(cells_to_remove)

    # set up agent generative model (same priors, same expectations)
    gm_full = make_generative_model(env_full, builder_generation=False)
    gm_broken = make_generative_model(env_broken, builder_generation=False)

    agent_full = make_agent(gm_full)
    agent_broken = make_agent(gm_broken)

    n_episodes = 20
    T = 30

    trajs_full = []
    trajs_broken = []

    for ep in range(n_episodes):
        trajs_full.append(run_episode(env_full, agent_full, T=T))
        trajs_broken.append(run_episode(env_broken, agent_broken, T=T))

    def stack(trajs, key):
        return np.stack([tr[key] for tr in trajs], axis=0)

    np.savez(
        outdir / "inference_from_absence_runs.npz",
        tile_map_full=env_full.tile_map,
        tile_map_broken=env_broken.tile_map,
        loc_full=stack(trajs_full, "loc"),
        loc_broken=stack(trajs_broken, "loc"),
        free_energy_full=stack(trajs_full, "free_energy"),
        free_energy_broken=stack(trajs_broken, "free_energy"),
        action_full=stack(trajs_full, "action"),
        action_broken=stack(trajs_broken, "action"),
    )

    print(f"Saved inference-from-absence logs to {outdir / 'inference_from_absence_runs.npz'}")

if __name__ == "__main__":
    main()
