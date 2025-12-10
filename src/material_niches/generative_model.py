# src/material_niches/generative_model.py

from typing import Dict, List, Tuple
import numpy as np
from pymdp import utils

from .env import NicheGridEnv, GridConfig, ACTION_MODIFY

def build_grid_indices(config: GridConfig) -> List[Tuple[int, int]]:
    """Return list of (y, x) tuples in linear index order."""
    coords = []
    for y in range(config.height):
        for x in range(config.width):
            coords.append((y, x))
    return coords

def build_A(env: NicheGridEnv) -> List[np.ndarray]:
    """
    Build A matrices for:
    - landmark modality (3 outcomes)
    - arousal modality (2 outcomes)
    """
    num_cells = env.num_cells
    num_tile_states = 2  # Wild / Marked
    num_hidden = num_cells * num_tile_states

    # landmark: [3, hidden_states]
    A_land = np.zeros((3, num_hidden))
    # arousal: [2, hidden_states]
    A_arousal = np.zeros((2, num_hidden))

    for loc_idx in range(num_cells):
        for tile_state in range(num_tile_states):
            hidden_idx = loc_idx * num_tile_states + tile_state

            x, y = env._idx_to_xy(loc_idx)
            is_goal = (y, x) in [(gy, gx) for (gy, gx) in env.config.goal_positions]

            # base: all zeros, then assign
            if is_goal:
                # GoalCue, high precision
                A_land[:, hidden_idx] = np.array([0.01, 0.01, 0.98])
                A_arousal[:, hidden_idx] = np.array([0.9, 0.1])  # mostly low
            elif tile_state == env.MARKED:
                # MarkedCue, high precision
                A_land[:, hidden_idx] = np.array([0.01, 0.98, 0.01])
                A_arousal[:, hidden_idx] = np.array([0.8, 0.2])
            else:
                # Wild: noisy / ambiguous
                A_land[:, hidden_idx] = np.array([0.5, 0.3, 0.2])
                A_arousal[:, hidden_idx] = np.array([0.3, 0.7])

    # ensure column-normalized
    A_land /= A_land.sum(axis=0, keepdims=True)
    A_arousal /= A_arousal.sum(axis=0, keepdims=True)

    return [A_land, A_arousal]

def build_B(env: NicheGridEnv) -> List[np.ndarray]:
    """
    Build transition matrices for:
    - location factor
    - tile factor
    """
    num_cells = env.num_cells
    num_tile_states = 2
    num_actions = 6  # up/down/left/right/stay/modify

    # B_loc: [num_cells, num_cells, num_actions]
    B_loc = np.zeros((num_cells, num_cells, num_actions))
    for prev_idx in range(num_cells):
        for action in range(num_actions):
            # clone env logic for movement
            x, y = env._idx_to_xy(prev_idx)
            new_x, new_y = x, y
            if action == 0:   # up
                new_y = max(0, y - 1)
            elif action == 1: # down
                new_y = min(env.config.height - 1, y + 1)
            elif action == 2: # left
                new_x = max(0, x - 1)
            elif action == 3: # right
                new_x = min(env.config.width - 1, x + 1)
            # stay or modify: no movement

            new_idx = env._xy_to_idx(new_x, new_y)
            B_loc[new_idx, prev_idx, action] = 1.0

    # B_tile: [2, 2, num_actions]
    B_tile = np.zeros((num_tile_states, num_tile_states, num_actions))
    for prev_tile in range(num_tile_states):
        for action in range(num_actions):
            if action == ACTION_MODIFY and prev_tile == env.WILD:
                B_tile[env.MARKED, prev_tile, action] = 1.0
            else:
                # identity
                B_tile[prev_tile, prev_tile, action] = 1.0

    return [B_loc, B_tile]

def build_C() -> List[np.ndarray]:
    """
    Observation preferences C:

    - Landmark modality: prefer GoalCue > MarkedCue > WildCue
    - Arousal modality: prefer Low arousal
    """
    C_land = np.array([0.0, 1.0, 4.0])  # log-preferences
    C_arousal = np.array([2.0, 0.0])    # low > high

    return [C_land, C_arousal]

def build_D(env: NicheGridEnv, builder_generation: bool) -> List[np.ndarray]:
    """
    Priors over hidden states:

    - D_loc: mass on start region
    - D_tile: depends on generation
    """
    num_cells = env.num_cells
    num_tile_states = 2

    D_loc = np.zeros(num_cells)
    for (y, x) in env.config.start_positions:
        idx = env._xy_to_idx(x, y)
        D_loc[idx] = 1.0
    D_loc /= D_loc.sum()

    # Builder gen: tiles expected to be Wild, Inheritor gen: more liberal priors
    if builder_generation:
        D_tile = np.array([0.9, 0.1])   # mostly Wild
    else:
        D_tile = np.array([0.4, 0.6])   # expect some Marked tiles

    return [D_loc, D_tile]

def make_generative_model(env: NicheGridEnv,
                          builder_generation: bool = True) -> Dict[str, List[np.ndarray]]:
    """Convenience wrapper."""
    A = build_A(env)
    B = build_B(env)
    C = build_C()
    D = build_D(env, builder_generation)
    return {"A": A, "B": B, "C": C, "D": D}
