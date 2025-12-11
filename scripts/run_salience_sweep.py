# scripts/run_salience_sweep.py

from pathlib import Path
import numpy as np

from material_niches.env import GridConfig, NicheGridEnv
from material_niches.generative_model import make_generative_model
from material_niches.agents import make_agent
from material_niches.simulate import run_episode

def simulate_with_salience(salience: float,
                           tile_map: np.ndarray,
                           n_episodes: int = 20,
                           T: int = 30):
    config = GridConfig(
        width=10,
        height=5,
        start_positions=[(4, 0)],
        goal_positions=[(2, 9)],
    )

    env = NicheGridEnv(config, initial_tile_map=tile_map)
    gm = make_generative_model(env, builder_generation=False, salience=salience)
    agent = make_agent(gm)

    trajs = []
    for ep in range(n_episodes):
        trajs.append(run_episode(env, agent, T=T))

    loc = np.stack([tr["loc"] for tr in trajs], axis=0)
    fe = np.stack([tr["free_energy"] for tr in trajs], axis=0)

    return loc, fe

def main():
    gen1_file = Path("results/gen1_builders/builder_runs.npz")
    if not gen1_file.exists():
        raise FileNotFoundError("Run Gen1 first to build a path")

    data = np.load(gen1_file, allow_pickle=True)
    tile_map = data["tile_map"]

    outdir = Path("results") / "salience_sweep"
    outdir.mkdir(parents=True, exist_ok=True)

    saliences = [0.5, 1.0, 2.0, 4.0]  # 0.5 ~ more diffuse, 4.0 ~ very sharp markers

    sweep_summary = []

    for s in saliences:
        loc, fe = simulate_with_salience(s, tile_map, n_episodes=20, T=30)
        # crude summary: mean FE across all timesteps and episodes
        mean_fe = float(fe.mean())
        sweep_summary.append((s, mean_fe))
        np.savez(
            outdir / f"salience_{s:.2f}.npz",
            salience=s,
            loc=loc,
            free_energy=fe,
        )
        print(f"Salience={s:.2f}: mean FE={mean_fe:.3f}")

    # Save sweep table
    np.savez(outdir / "summary.npz", summary=np.array(sweep_summary, dtype=float))
    print(f"Saved salience sweep results to {outdir}")

if __name__ == "__main__":
    main()
