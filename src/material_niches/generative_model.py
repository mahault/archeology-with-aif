# src/material_niches/generative_model.py

from typing import Dict, List
import numpy as np
from pymdp import utils

from .env import NicheGridEnv, GridConfig, ACTION_MODIFY

def _sharpen_probs(p: np.ndarray, salience: float) -> np.ndarray:
    """
    Raise probabilities to a power and renormalize.
    salience > 1 -> sharper / more peaked.
    salience = 1 -> unchanged.
    """
    p = np.asarray(p, dtype=float)
    p = p ** salience
    p = p / p.sum()
    return p

def build_A(env: NicheGridEnv, salience: float = 1.0) -> List[np.ndarray]:
    """
    Build A matrices for:
    - landmark modality (3 outcomes)
    - arousal modality (2 outcomes)

    SHAPE CONVENTION (pymdp v1.0.0_alpha):

    For F hidden state factors with dimensions `num_states = [N1, N2, ...]`,
    and an observation modality m that depends on *all* factors, we must have:

        A[m].shape == (batch_size, n_obs_m, N1, N2, ..., NF)

    Our model has:
      - factor 0: location, size N_loc = env.num_cells
      - factor 1: tile type, size 2 (Wild / Marked)

    So we want:
      - A_land   : (1, 3, N_loc, 2)
      - A_arousal: (1, 2, N_loc, 2)
    """

    num_cells = env.num_cells
    N_loc = num_cells
    num_tile_states = 2  # Wild / Marked

    # base distributions (before sharpening)
    wild_land = np.array([0.5, 0.3, 0.2])
    marked_land = np.array([0.01, 0.98, 0.01])
    goal_land = np.array([0.01, 0.01, 0.98])

    wild_arousal = np.array([0.3, 0.7])   # low, high
    marked_arousal = np.array([0.8, 0.2])
    goal_arousal = np.array([0.9, 0.1])

    # apply salience to landmark distributions (not to arousal for now)
    wild_land_s = _sharpen_probs(wild_land, 1.0)         # keep wild noisy
    marked_land_s = _sharpen_probs(marked_land, salience)
    goal_land_s = _sharpen_probs(goal_land, salience)

    # First build factorised 3D A:
    #   A_land_base   : (3, N_loc, 2)
    #   A_arousal_base: (2, N_loc, 2)
    A_land_base = np.zeros((3, N_loc, num_tile_states))
    A_arousal_base = np.zeros((2, N_loc, num_tile_states))

    for loc_idx in range(N_loc):
        x, y = env._idx_to_xy(loc_idx)
        is_goal = (y, x) in env.config.goal_positions

        for tile_state in range(num_tile_states):
            if is_goal:
                A_land_base[:, loc_idx, tile_state] = goal_land_s
                A_arousal_base[:, loc_idx, tile_state] = goal_arousal
            elif tile_state == env.MARKED:
                A_land_base[:, loc_idx, tile_state] = marked_land_s
                A_arousal_base[:, loc_idx, tile_state] = marked_arousal
            else:
                # Wild
                A_land_base[:, loc_idx, tile_state] = wild_land_s
                A_arousal_base[:, loc_idx, tile_state] = wild_arousal

    # Normalize over observation dimension (axis=0) for each (loc, tile)
    A_land_base /= A_land_base.sum(axis=0, keepdims=True)
    A_arousal_base /= A_arousal_base.sum(axis=0, keepdims=True)

    # Add batch dimension at axis=0 so shapes become (1, n_obs, N_loc, 2)
    A_land = A_land_base[None, ...]
    A_arousal = A_arousal_base[None, ...]

    return [A_land, A_arousal]



def build_B(env: NicheGridEnv) -> List[np.ndarray]:
    """
    Build transition matrices for:
    - location factor
    - tile factor

    SHAPE CONVENTION (pymdp JAX):
    B[f].shape == (batch_size, next_state, prev_state, num_actions)
    """
    num_cells = env.num_cells
    num_tile_states = 2
    num_actions = 6  # up/down/left/right/stay/modify

    # B_loc_base: [num_cells, num_cells, num_actions] -> (next, prev, action)
    B_loc_base = np.zeros((num_cells, num_cells, num_actions))
    for prev_idx in range(num_cells):
        for action in range(num_actions):
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
            B_loc_base[new_idx, prev_idx, action] = 1.0

    # B_tile_base: [2, 2, num_actions]
    B_tile_base = np.zeros((num_tile_states, num_tile_states, num_actions))
    for prev_tile in range(num_tile_states):
        for action in range(num_actions):
            if action == ACTION_MODIFY and prev_tile == env.WILD:
                B_tile_base[env.MARKED, prev_tile, action] = 1.0
            else:
                B_tile_base[prev_tile, prev_tile, action] = 1.0

    # Add batch dimension at axis=0
    B_loc = B_loc_base[None, ...]     # (1, num_cells, num_cells, num_actions)
    B_tile = B_tile_base[None, ...]   # (1, 2, 2, num_actions)

    return [B_loc, B_tile]

def build_C() -> List[np.ndarray]:
    """
    Observation preferences C:

    - Landmark modality: prefer GoalCue > MarkedCue > WildCue
    - Arousal modality: prefer Low arousal

    SHAPE CONVENTION (pymdp JAX):
    C[m].shape == (batch_size, num_obs_m)
    """
    C_land = np.array([[0.0, 1.0, 4.0]])  # log-preferences, shape (1, 3)
    C_arousal = np.array([[2.0, 0.0]])    # low > high, shape (1, 2)

    return [C_land, C_arousal]

def build_D(env: NicheGridEnv, builder_generation: bool) -> List[np.ndarray]:
    """
    Priors over hidden states:

    - D_loc: mass on start region
    - D_tile: depends on generation

    SHAPE CONVENTION (pymdp JAX):
    D[f].shape == (batch_size, num_states_f)
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

    # Add batch dimension
    D_loc = D_loc[None, ...]      # (1, num_cells)
    D_tile = D_tile[None, ...]    # (1, 2)

    return [D_loc, D_tile]

def make_generative_model(
    env: NicheGridEnv,
    builder_generation: bool = True,
    salience: float = 1.0,
) -> Dict[str, List[np.ndarray]]:
    """Convenience wrapper."""
    A = build_A(env, salience=salience)
    B = build_B(env)
    C = build_C()
    D = build_D(env, builder_generation)
    return {"A": A, "B": B, "C": C, "D": D}
