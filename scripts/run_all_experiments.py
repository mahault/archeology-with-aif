# scripts/run_all_experiments.py

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from material_niches.env import GridConfig, NicheGridEnv
from material_niches.generative_model import make_generative_model
from material_niches.agents import make_agent
from material_niches.simulate import run_generation, run_episode
from material_niches.analysis import (
    plot_path_heatmap,
    plot_free_energy,
    basic_summary,
)


def run_gen1_builders(config: GridConfig,
                      outdir: Path,
                      n_episodes: int = 20,
                      T: int = 30,
                      seed: int = 42):
    outdir.mkdir(parents=True, exist_ok=True)

    env, trajectories = run_generation(
        config,
        builder_generation=True,
        n_episodes=n_episodes,
        T=T,
        seed=seed,
    )

    loc = np.stack([traj["loc"] for traj in trajectories], axis=0)
    tile = np.stack([traj["tile"] for traj in trajectories], axis=0)
    action = np.stack([traj["action"] for traj in trajectories], axis=0)
    free_energy = np.stack([traj["free_energy"] for traj in trajectories], axis=0)

    np.savez(
        outdir / "builder_runs.npz",
        tile_map=env.tile_map,
        loc=loc,
        tile=tile,
        action=action,
        free_energy=free_energy,
    )

    print(f"[Gen1] Saved logs to {outdir / 'builder_runs.npz'}")

    # Plots
    plot_path_heatmap(env.tile_map, config.width, config.height,
                      title="Gen1 final tile map (builders)")
    plt.savefig(outdir / "gen1_tile_map.png", dpi=150)
    plt.close()

    plot_free_energy(free_energy, title="Gen1 free energy (builders)")
    plt.savefig(outdir / "gen1_free_energy.png", dpi=150)
    plt.close()

    summary = basic_summary(loc, free_energy)
    print("[Gen1] Summary:", summary)

    return env.tile_map, summary


def run_gen2_inheritors(config: GridConfig,
                        builder_tile_map: np.ndarray,
                        outdir: Path,
                        n_episodes: int = 20,
                        T: int = 30,
                        seed: int = 123):
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    env = NicheGridEnv(config, rng=rng, initial_tile_map=builder_tile_map)
    gm = make_generative_model(env, builder_generation=False)
    agent = make_agent(gm)

    trajectories = []
    for ep in range(n_episodes):
        traj = run_episode(env, agent, T=T)
        trajectories.append(traj)

    loc = np.stack([traj["loc"] for traj in trajectories], axis=0)
    tile = np.stack([traj["tile"] for traj in trajectories], axis=0)
    action = np.stack([traj["action"] for traj in trajectories], axis=0)
    free_energy = np.stack([traj["free_energy"] for traj in trajectories], axis=0)

    np.savez(
        outdir / "inheritor_runs.npz",
        tile_map=env.tile_map,
        loc=loc,
        tile=tile,
        action=action,
        free_energy=free_energy,
    )

    print(f"[Gen2] Saved logs to {outdir / 'inheritor_runs.npz'}")

    # Plots
    plot_path_heatmap(env.tile_map, config.width, config.height,
                      title="Gen2 tile map (inheritors)")
    plt.savefig(outdir / "gen2_tile_map.png", dpi=150)
    plt.close()

    plot_free_energy(free_energy, title="Gen2 free energy (inheritors)")
    plt.savefig(outdir / "gen2_free_energy.png", dpi=150)
    plt.close()

    summary = basic_summary(loc, free_energy)
    print("[Gen2] Summary:", summary)

    return summary


def run_inference_from_absence(config: GridConfig,
                               builder_tile_map: np.ndarray,
                               outdir: Path,
                               n_episodes: int = 20,
                               T: int = 30,
                               seed: int = 999):
    outdir.mkdir(parents=True, exist_ok=True)

    # Full path environment
    env_full = NicheGridEnv(config, initial_tile_map=builder_tile_map)

    # Broken path environment (taphonomy)
    env_broken = env_full.clone_with_tilemap(builder_tile_map)
    # remove a middle segment of the corridor on row y=2, x=4..6
    cells_to_remove = [(2, x) for x in range(4, 7)]
    env_broken.remove_path_segment(cells_to_remove)

    gm_full = make_generative_model(env_full, builder_generation=False)
    gm_broken = make_generative_model(env_broken, builder_generation=False)

    agent_full = make_agent(gm_full)
    agent_broken = make_agent(gm_broken)

    rng = np.random.default_rng(seed)

    trajs_full = []
    trajs_broken = []

    for ep in range(n_episodes):
        # to keep some stochasticity, we could re-seed envs each episode if needed
        trajs_full.append(run_episode(env_full, agent_full, T=T))
        trajs_broken.append(run_episode(env_broken, agent_broken, T=T))

    def stack(trajs, key):
        return np.stack([tr[key] for tr in trajs], axis=0)

    loc_full = stack(trajs_full, "loc")
    loc_broken = stack(trajs_broken, "loc")
    fe_full = stack(trajs_full, "free_energy")
    fe_broken = stack(trajs_broken, "free_energy")

    np.savez(
        outdir / "inference_from_absence_runs.npz",
        tile_map_full=env_full.tile_map,
        tile_map_broken=env_broken.tile_map,
        loc_full=loc_full,
        loc_broken=loc_broken,
        free_energy_full=fe_full,
        free_energy_broken=fe_broken,
    )

    print(f"[Absence] Saved logs to {outdir / 'inference_from_absence_runs.npz'}")

    # Plots: tile maps
    plot_path_heatmap(env_full.tile_map, config.width, config.height,
                      title="Intact path (full)")
    plt.savefig(outdir / "tile_map_full.png", dpi=150)
    plt.close()

    plot_path_heatmap(env_broken.tile_map, config.width, config.height,
                      title="Broken path (taphonomy)")
    plt.savefig(outdir / "tile_map_broken.png", dpi=150)
    plt.close()

    # Plots: FE trajectories
    plot_free_energy(fe_full, title="Free energy (intact path)")
    plt.savefig(outdir / "fe_full.png", dpi=150)
    plt.close()

    plot_free_energy(fe_broken, title="Free energy (broken path)")
    plt.savefig(outdir / "fe_broken.png", dpi=150)
    plt.close()

    # Simple overlay of mean FE full vs broken
    mean_fe_full = fe_full.mean(axis=0)
    mean_fe_broken = fe_broken.mean(axis=0)
    t = np.arange(mean_fe_full.shape[0])

    plt.figure()
    plt.plot(t, mean_fe_full, label="Intact path")
    plt.plot(t, mean_fe_broken, label="Broken path")
    plt.xlabel("time")
    plt.ylabel("free energy (mean)")
    plt.title("Free energy: intact vs broken path")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fe_full_vs_broken.png", dpi=150)
    plt.close()

    print("[Absence] Mean FE (full):   ", float(mean_fe_full.mean()))
    print("[Absence] Mean FE (broken): ", float(mean_fe_broken.mean()))

    return {
        "mean_FE_full": float(mean_fe_full.mean()),
        "mean_FE_broken": float(mean_fe_broken.mean()),
    }


def run_salience_sweep(config: GridConfig,
                       builder_tile_map: np.ndarray,
                       outdir: Path,
                       saliences=None,
                       n_episodes: int = 20,
                       T: int = 30):
    if saliences is None:
        saliences = [0.5, 1.0, 2.0, 4.0]

    outdir.mkdir(parents=True, exist_ok=True)

    results = []

    for s in saliences:
        env = NicheGridEnv(config, initial_tile_map=builder_tile_map)
        gm = make_generative_model(env, builder_generation=False, salience=s)
        agent = make_agent(gm)

        trajs = []
        for ep in range(n_episodes):
            trajs.append(run_episode(env, agent, T=T))

        loc = np.stack([tr["loc"] for tr in trajs], axis=0)
        fe = np.stack([tr["free_energy"] for tr in trajs], axis=0)

        mean_fe = float(fe.mean())
        results.append((s, mean_fe))

        np.savez(
            outdir / f"salience_{s:.2f}.npz",
            salience=s,
            loc=loc,
            free_energy=fe,
        )
        print(f"[Salience] s={s:.2f}: mean FE={mean_fe:.3f}")

    results = np.array(results, dtype=float)
    np.savez(outdir / "summary.npz", summary=results)

    # Plot mean FE vs salience
    sal = results[:, 0]
    meanfe = results[:, 1]

    plt.figure()
    plt.plot(sal, meanfe, marker="o")
    plt.xlabel("salience")
    plt.ylabel("mean free energy")
    plt.title("Effect of monument salience on free energy")
    plt.tight_layout()
    plt.savefig(outdir / "salience_vs_meanFE.png", dpi=150)
    plt.close()

    return results


def main():
    base_results = Path("results")

    # Shared geometry: simple corridor
    config = GridConfig(
        width=10,
        height=5,
        start_positions=[(4, 0)],  # (y, x)
        goal_positions=[(2, 9)],
    )

    # 1. Gen1 builders
    gen1_dir = base_results / "gen1_builders"
    builder_tile_map, gen1_summary = run_gen1_builders(
        config=config,
        outdir=gen1_dir,
        n_episodes=20,
        T=30,
        seed=42,
    )

    # 2. Gen2 inheritors
    gen2_dir = base_results / "gen2_inheritors"
    gen2_summary = run_gen2_inheritors(
        config=config,
        builder_tile_map=builder_tile_map,
        outdir=gen2_dir,
        n_episodes=20,
        T=30,
        seed=123,
    )

    # 3. Inference from absence
    absence_dir = base_results / "inference_from_absence"
    absence_summary = run_inference_from_absence(
        config=config,
        builder_tile_map=builder_tile_map,
        outdir=absence_dir,
        n_episodes=20,
        T=30,
        seed=999,
    )

    # 4. Salience sweep
    salience_dir = base_results / "salience_sweep"
    salience_summary = run_salience_sweep(
        config=config,
        builder_tile_map=builder_tile_map,
        outdir=salience_dir,
        saliences=[0.5, 1.0, 2.0, 4.0],
        n_episodes=20,
        T=30,
    )

    # Quick global summary print
    print("\n=== Global Summary ===")
    print("Gen1 (builders):", gen1_summary)
    print("Gen2 (inheritors):", gen2_summary)
    print("Inference from absence:", absence_summary)
    print("Salience sweep (salience, mean_FE):", salience_summary)


if __name__ == "__main__":
    main()
