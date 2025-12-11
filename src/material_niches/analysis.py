# src/material_niches/analysis.py

from pathlib import Path
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt

def load_npz(path: str | Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}

def plot_path_heatmap(tile_map: np.ndarray, width: int, height: int, title: str = ""):
    """
    Visualize a tile_map (0=Wild, 1=Marked) as a 2D heatmap.
    """
    grid = tile_map.reshape(height, width)
    plt.figure()
    plt.imshow(grid, origin="lower", interpolation="nearest")
    plt.colorbar(label="Tile type (0=Wild, 1=Marked)")
    plt.title(title or "Tile map")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

def plot_free_energy(fe_traj: np.ndarray, title: str = ""):
    """
    fe_traj: shape [episodes, T]
    """
    mean_fe = fe_traj.mean(axis=0)
    std_fe = fe_traj.std(axis=0)

    t = np.arange(mean_fe.shape[0])
    plt.figure()
    plt.plot(t, mean_fe, label="mean FE")
    plt.fill_between(t, mean_fe - std_fe, mean_fe + std_fe, alpha=0.3)
    plt.xlabel("time")
    plt.ylabel("free energy (approx)")
    plt.title(title or "Free Energy over time")
    plt.legend()
    plt.tight_layout()

def basic_summary(loc: np.ndarray, free_energy: np.ndarray) -> Dict[str, float]:
    """
    loc: [episodes, T]
    free_energy: [episodes, T]
    """
    episodes, T = loc.shape
    avg_steps = T  # we don't have an explicit 'done' flag yet

    return {
        "episodes": episodes,
        "horizon": T,
        "mean_FE": float(free_energy.mean()),
        "std_FE": float(free_energy.std()),
        "avg_steps": avg_steps,
    }
