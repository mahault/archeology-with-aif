# scripts/run_all_experiments.py

from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt

from material_niches.env import GridConfig, NicheGridEnv
from material_niches.generative_model import make_generative_model
from material_niches.agents import make_agent
from material_niches.simulate import run_generation, run_episode, run_generation_with_learning
from material_niches.analysis import (
    plot_path_heatmap,
    plot_free_energy,
    basic_summary,
)


def setup_logging(log_dir: Path):
    """Setup logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging to {log_file}")
    return log_file


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
    agent = make_agent(gm, policy_len=3)

    import jax.random as jr
    rng_key = jr.PRNGKey(seed)

    trajectories = []
    for ep in range(n_episodes):
        rng_key, ep_key = jr.split(rng_key)
        verbose = (ep == 0 or ep == n_episodes - 1)  # Log first and last
        traj = run_episode(env, agent, T=T, episode_num=ep, rng_key=ep_key, verbose=verbose)
        trajectories.append(traj)
        if ep % 5 == 0:
            print(f"[Gen2] Episode {ep+1}/{n_episodes}")

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

    agent_full = make_agent(gm_full, policy_len=3)
    agent_broken = make_agent(gm_broken, policy_len=3)

    import jax.random as jr
    rng_key = jr.PRNGKey(seed)

    trajs_full = []
    trajs_broken = []

    for ep in range(n_episodes):
        rng_key, key1, key2 = jr.split(rng_key, 3)
        verbose = (ep == 0 or ep == n_episodes - 1)
        trajs_full.append(run_episode(env_full, agent_full, T=T, episode_num=ep, rng_key=key1, verbose=verbose))
        trajs_broken.append(run_episode(env_broken, agent_broken, T=T, episode_num=ep, rng_key=key2, verbose=verbose))
        if ep % 5 == 0:
            print(f"[Absence] Episode {ep+1}/{n_episodes}")

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

    import jax.random as jr
    rng_key = jr.PRNGKey(0)

    for s in saliences:
        env = NicheGridEnv(config, initial_tile_map=builder_tile_map)
        gm = make_generative_model(env, builder_generation=False, salience=s)
        agent = make_agent(gm, policy_len=3)

        trajs = []
        for ep in range(n_episodes):
            rng_key, ep_key = jr.split(rng_key)
            verbose = (ep == 0)
            trajs.append(run_episode(env, agent, T=T, episode_num=ep, rng_key=ep_key, verbose=verbose))

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


def run_learning_experiment(config: GridConfig,
                            outdir: Path,
                            n_episodes: int = 20,
                            T: int = 50,
                            seed: int = 42,
                            num_waypoints: int = 4,
                            waypoint_strength: float = 2.0,
                            action_selection: str = "stochastic"):
    """
    Run experiment with learning and hierarchical planning.

    This tests the new system where:
    1. Agent builds a map of explored vs unexplored areas
    2. Agent follows waypoints toward the goal
    3. Epistemic value decreases for explored areas
    """
    outdir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 60)
    logging.info("LEARNING EXPERIMENT (with hierarchical planning)")
    logging.info(f"  waypoints={num_waypoints}, waypoint_strength={waypoint_strength}")
    logging.info("=" * 60)

    env, trajectories, world_map = run_generation_with_learning(
        config,
        builder_generation=True,
        n_episodes=n_episodes,
        T=T,
        seed=seed,
        num_waypoints=num_waypoints,
        waypoint_strength=waypoint_strength,
        reset_map_each_episode=True,  # Each episode agent starts fresh
        action_selection=action_selection,
    )

    # Stack trajectories
    loc = np.stack([traj["loc"] for traj in trajectories], axis=0)
    tile = np.stack([traj["tile"] for traj in trajectories], axis=0)
    action = np.stack([traj["action"] for traj in trajectories], axis=0)
    free_energy = np.stack([traj["free_energy"] for traj in trajectories], axis=0)

    # Check how many times goal was reached
    goal_y, goal_x = config.goal_positions[0]
    goal_idx = env._xy_to_idx(goal_x, goal_y)
    goals_reached = sum(1 for traj in trajectories if traj["loc"][-1] == goal_idx)

    np.savez(
        outdir / "learning_runs.npz",
        tile_map=env.tile_map,
        loc=loc,
        tile=tile,
        action=action,
        free_energy=free_energy,
        goals_reached=goals_reached,
    )

    print(f"[Learning] Saved logs to {outdir / 'learning_runs.npz'}")

    # Plots
    plot_path_heatmap(env.tile_map, config.width, config.height,
                      title="Learning experiment: final tile map")
    plt.savefig(outdir / "learning_tile_map.png", dpi=150)
    plt.close()

    plot_free_energy(free_energy, title="Learning experiment: free energy")
    plt.savefig(outdir / "learning_free_energy.png", dpi=150)
    plt.close()

    # Plot trajectory heatmap (where did the agent go most often?)
    visit_counts = np.zeros(env.num_cells)
    for traj in trajectories:
        for loc_idx in traj["loc"]:
            visit_counts[loc_idx] += 1

    visit_map = visit_counts.reshape(config.height, config.width)
    plt.figure(figsize=(10, 5))
    plt.imshow(visit_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Visit count')
    plt.title('Agent trajectory heatmap (learning experiment)')

    # Mark start and goal
    start_y, start_x = config.start_positions[0]
    goal_y, goal_x = config.goal_positions[0]
    plt.scatter([start_x], [start_y], c='green', s=200, marker='s', label='Start')
    plt.scatter([goal_x], [goal_y], c='blue', s=200, marker='*', label='Goal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "learning_trajectory_heatmap.png", dpi=150)
    plt.close()

    summary = basic_summary(loc, free_energy)
    summary['goals_reached'] = goals_reached
    summary['goal_rate'] = goals_reached / n_episodes
    print(f"[Learning] Summary: {summary}")
    logging.info(f"[Learning] Goals reached: {goals_reached}/{n_episodes} ({100*goals_reached/n_episodes:.1f}%)")

    return env.tile_map, summary


def main():
    base_results = Path("results")
    log_dir = base_results / "logs"

    # Setup logging to file and console
    log_file = setup_logging(log_dir)
    logging.info("=" * 60)
    logging.info("STARTING EXPERIMENT RUN")
    logging.info("=" * 60)

    # Shared geometry: simple corridor
    config = GridConfig(
        width=10,
        height=5,
        start_positions=[(4, 0)],  # (y, x)
        goal_positions=[(2, 9)],
    )

    # 0. NEW: Learning experiment with hierarchical planning
    learning_dir = base_results / "learning_experiment"
    learning_tile_map, learning_summary = run_learning_experiment(
        config=config,
        outdir=learning_dir,
        n_episodes=20,
        T=50,  # Longer episodes to reach goal
        seed=42,
        num_waypoints=4,
        waypoint_strength=4.0,  # Increased from 2.0 to overcome comfort trap (C_land+C_arousal=2.5)
    )

    # 1. Gen1 builders (original method for comparison)
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
    logging.info("=" * 60)
    logging.info("GLOBAL SUMMARY")
    logging.info("=" * 60)
    logging.info(f"LEARNING (hierarchical): {learning_summary}")
    logging.info(f"Gen1 (builders): {gen1_summary}")
    logging.info(f"Gen2 (inheritors): {gen2_summary}")
    logging.info(f"Inference from absence: {absence_summary}")
    logging.info(f"Salience sweep (salience, mean_FE): {salience_summary.tolist()}")
    logging.info("=" * 60)
    logging.info(f"EXPERIMENT COMPLETE - Log saved to {log_file}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
