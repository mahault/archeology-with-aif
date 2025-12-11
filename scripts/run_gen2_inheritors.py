# scripts/run_gen2_inheritors.py

from pathlib import Path
import numpy as np

from material_niches.env import GridConfig, NicheGridEnv
from material_niches.simulate import run_generation

def main():
    gen1_file = Path("results/gen1_builders/builder_runs.npz")
    if not gen1_file.exists():
        raise FileNotFoundError("Run Gen1 first: results/gen1_builders/builder_runs.npz not found")

    data = np.load(gen1_file, allow_pickle=True)
    builder_tile_map = data["tile_map"]

    outdir = Path("results") / "gen2_inheritors"
    outdir.mkdir(parents=True, exist_ok=True)

    # same geometry as Gen1
    config = GridConfig(
        width=10,
        height=5,
        start_positions=[(4, 0)],
        goal_positions=[(2, 9)],
    )

    # Create env with inherited path
    base_env = NicheGridEnv(config, initial_tile_map=builder_tile_map)

    # run_generation makes its own env internally, so we pass the config & override
    # Here we'll just re-create a matching env inside, using the inherited tile_map
    # by monkey-patching run_generation style: simplest is to write a small local loop.

    from material_niches.generative_model import make_generative_model
    from material_niches.agents import make_agent
    from material_niches.simulate import run_episode

    import numpy as np

    rng = np.random.default_rng(123)
    env = NicheGridEnv(config, rng=rng, initial_tile_map=builder_tile_map)
    gm = make_generative_model(env, builder_generation=False)
    agent = make_agent(gm)

    n_episodes = 20
    T = 30
    trajectories = []

    for ep in range(n_episodes):
        traj = run_episode(env, agent, T=T)
        trajectories.append(traj)

    arr_loc = np.stack([traj["loc"] for traj in trajectories], axis=0)
    arr_tile = np.stack([traj["tile"] for traj in trajectories], axis=0)
    arr_act = np.stack([traj["action"] for traj in trajectories], axis=0)
    arr_fe = np.stack([traj["free_energy"] for traj in trajectories], axis=0)

    np.savez(
        outdir / "inheritor_runs.npz",
        tile_map=env.tile_map,
        loc=arr_loc,
        tile=arr_tile,
        action=arr_act,
        free_energy=arr_fe,
    )

    print(f"Saved inheritor generation logs to {outdir / 'inheritor_runs.npz'}")

if __name__ == "__main__":
    main()
