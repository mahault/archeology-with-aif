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
    - landmark modality (3 outcomes): Wild, Marked, Goal
    - arousal modality (2 outcomes): Low, High
    - position modality (N_loc outcomes): Direct position observation

    The position modality creates epistemic value for exploration:
    - Each location gives a unique observation
    - Agent is uncertain about what it will observe at unvisited locations
    - This drives exploration through information gain (epistemic value)

    SHAPE CONVENTION (pymdp v1.0.0_alpha):
        A[m].shape == (batch_size, n_obs_m, N1, N2, ..., NF)
    """

    num_cells = env.num_cells
    N_loc = num_cells
    num_tile_states = 2  # Wild / Marked

    # === Landmark modality (3 outcomes) ===
    # Deterministic mapping based on tile type and goal location
    A_land_base = np.zeros((3, N_loc, num_tile_states))

    for loc_idx in range(N_loc):
        x, y = env._idx_to_xy(loc_idx)
        is_goal = (y, x) in env.config.goal_positions

        for tile_state in range(num_tile_states):
            if is_goal:
                # Goal location: always gives GoalCue
                A_land_base[:, loc_idx, tile_state] = [0.01, 0.01, 0.98]
            elif tile_state == env.MARKED:
                # Marked tile: gives MarkedCue
                A_land_base[:, loc_idx, tile_state] = [0.05, 0.93, 0.02]
            else:
                # Wild tile: gives WildCue
                A_land_base[:, loc_idx, tile_state] = [0.90, 0.08, 0.02]

    # Apply salience sharpening
    for loc_idx in range(N_loc):
        for tile_state in range(num_tile_states):
            A_land_base[:, loc_idx, tile_state] = _sharpen_probs(
                A_land_base[:, loc_idx, tile_state], salience
            )

    # === Arousal modality (2 outcomes) ===
    # Low arousal at marked/goal, high at wild
    A_arousal_base = np.zeros((2, N_loc, num_tile_states))

    for loc_idx in range(N_loc):
        x, y = env._idx_to_xy(loc_idx)
        is_goal = (y, x) in env.config.goal_positions

        for tile_state in range(num_tile_states):
            if is_goal:
                A_arousal_base[:, loc_idx, tile_state] = [0.95, 0.05]
            elif tile_state == env.MARKED:
                A_arousal_base[:, loc_idx, tile_state] = [0.80, 0.20]
            else:
                A_arousal_base[:, loc_idx, tile_state] = [0.30, 0.70]

    # === Position modality (N_loc outcomes) - for exploration ===
    # Key insight: epistemic value requires UNCERTAINTY that gets resolved by visiting.
    #
    # Design: Tile type affects position certainty
    # - MARKED tiles are "landmarks" → high certainty about position (you know where you are)
    # - WILD tiles are "uncharted" → low certainty (confusing, could be anywhere unexplored)
    #
    # This creates:
    # 1. Epistemic value for exploring wild areas (resolves uncertainty)
    # 2. Incentive to mark tiles (creates landmarks for navigation)
    # 3. Differentiation between policies that explore vs stay on marked paths

    A_pos_base = np.zeros((N_loc, N_loc, num_tile_states))

    for loc_idx in range(N_loc):
        x, y = env._idx_to_xy(loc_idx)

        # Get neighboring positions for locality-based uncertainty
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.config.width and 0 <= ny < env.config.height:
                neighbors.append(env._xy_to_idx(nx, ny))

        for tile_state in range(num_tile_states):
            if tile_state == env.MARKED:
                # MARKED = landmark: high certainty about position
                A_pos_base[loc_idx, loc_idx, tile_state] = 0.85
                # Small spread to neighbors
                neighbor_prob = 0.10 / max(len(neighbors), 1)
                for n_idx in neighbors:
                    A_pos_base[n_idx, loc_idx, tile_state] = neighbor_prob
                # Tiny spread to rest
                remaining = 0.05 / max(N_loc - 1 - len(neighbors), 1)
                for other_idx in range(N_loc):
                    if other_idx != loc_idx and other_idx not in neighbors:
                        A_pos_base[other_idx, loc_idx, tile_state] = remaining
            else:
                # WILD = uncharted: LOW certainty - this creates epistemic value!
                # Agent is uncertain about position in wild areas
                A_pos_base[loc_idx, loc_idx, tile_state] = 0.25
                # Higher spread to neighbors (locality preserved but uncertain)
                neighbor_prob = 0.35 / max(len(neighbors), 1)
                for n_idx in neighbors:
                    A_pos_base[n_idx, loc_idx, tile_state] = neighbor_prob
                # Significant spread to all other locations (genuine uncertainty)
                remaining = 0.40 / max(N_loc - 1 - len(neighbors), 1)
                for other_idx in range(N_loc):
                    if other_idx != loc_idx and other_idx not in neighbors:
                        A_pos_base[other_idx, loc_idx, tile_state] = remaining

    # Normalize
    A_land_base /= A_land_base.sum(axis=0, keepdims=True)
    A_arousal_base /= A_arousal_base.sum(axis=0, keepdims=True)
    A_pos_base /= A_pos_base.sum(axis=0, keepdims=True)

    # Add batch dimension
    A_land = A_land_base[None, ...]
    A_arousal = A_arousal_base[None, ...]
    A_pos = A_pos_base[None, ...]

    return [A_land, A_arousal, A_pos]



def build_B(env: NicheGridEnv) -> List[np.ndarray]:
    """
    Build transition matrices for:
    - location factor (controlled)
    - tile factor (uncontrolled - changes as side effect of MODIFY action)

    SHAPE CONVENTION (pymdp JAX):
    B[f].shape == (batch_size, next_state, prev_state, num_actions)

    For uncontrolled factors, we use num_actions=1 (identity transition).
    The tile change from MODIFY is modeled in the A matrix expectations.
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

    # B_tile_base: Tile transitions depend on the location action
    # This allows the agent to plan tile modifications
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

def build_C(env: NicheGridEnv) -> List[np.ndarray]:
    """
    Observation preferences C:

    - Landmark modality: prefer GoalCue > MarkedCue > WildCue
    - Arousal modality: prefer Low arousal
    - Position modality: no strong preference (exploration is driven by epistemic value)

    SHAPE CONVENTION (pymdp JAX):
    C[m].shape == (batch_size, num_obs_m)
    """
    num_cells = env.num_cells

    # Landmark: strong preference for goal
    C_land = np.array([[0.0, 0.5, 8.0]])  # log-preferences: Goal >> Marked > Wild

    # Arousal: prefer low arousal
    C_arousal = np.array([[2.0, 0.0]])    # low > high

    # Position: neutral (no preference for specific locations)
    # Exploration is driven by epistemic value (information gain), not by C
    C_pos = np.zeros((1, num_cells))

    return [C_land, C_arousal, C_pos]

def build_D(env: NicheGridEnv, builder_generation: bool) -> List[np.ndarray]:
    """
    Priors over hidden states:

    - D_loc: mass on start region (with smoothing to allow observation updates)
    - D_tile: depends on generation

    SHAPE CONVENTION (pymdp JAX):
    D[f].shape == (batch_size, num_states_f)
    """
    num_cells = env.num_cells
    num_tile_states = 2

    # D_loc: Strong prior on start, but with smoothing to allow belief updates
    # Without smoothing, point mass prior can't be updated by observations
    # (because 0 * likelihood = 0 for all non-start positions)
    D_loc = np.full(num_cells, 0.001)  # Small uniform baseline
    for (y, x) in env.config.start_positions:
        idx = env._xy_to_idx(x, y)
        D_loc[idx] = 0.95  # Strong but not 100%
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
    C = build_C(env)
    D = build_D(env, builder_generation)
    return {"A": A, "B": B, "C": C, "D": D}


# ============================================================================
# DYNAMIC MODEL UPDATES (for learning and hierarchical planning)
# ============================================================================

def build_A_with_exploration(env: NicheGridEnv, world_map, salience: float = 1.0) -> List[np.ndarray]:
    """
    Build A matrices where position uncertainty reflects ACTUAL exploration state.

    Key insight: epistemic value should come from real uncertainty about unexplored
    areas, not artificial noise.

    - Unexplored cells: HIGH uncertainty (agent doesn't know what's there)
    - Explored cells: LOW uncertainty (agent has learned what's there)

    This creates natural exploration → exploitation behavior.
    """
    num_cells = env.num_cells
    N_loc = num_cells
    num_tile_states = 2

    # === Landmark and Arousal modalities (same as before) ===
    A_land_base = np.zeros((3, N_loc, num_tile_states))
    A_arousal_base = np.zeros((2, N_loc, num_tile_states))

    for loc_idx in range(N_loc):
        x, y = env._idx_to_xy(loc_idx)
        is_goal = (y, x) in env.config.goal_positions

        for tile_state in range(num_tile_states):
            # Landmark
            if is_goal:
                A_land_base[:, loc_idx, tile_state] = [0.01, 0.01, 0.98]
            elif tile_state == env.MARKED:
                A_land_base[:, loc_idx, tile_state] = [0.05, 0.93, 0.02]
            else:
                A_land_base[:, loc_idx, tile_state] = [0.90, 0.08, 0.02]

            # Arousal
            if is_goal:
                A_arousal_base[:, loc_idx, tile_state] = [0.95, 0.05]
            elif tile_state == env.MARKED:
                A_arousal_base[:, loc_idx, tile_state] = [0.80, 0.20]
            else:
                A_arousal_base[:, loc_idx, tile_state] = [0.30, 0.70]

    # Apply salience
    for loc_idx in range(N_loc):
        for tile_state in range(num_tile_states):
            A_land_base[:, loc_idx, tile_state] = _sharpen_probs(
                A_land_base[:, loc_idx, tile_state], salience
            )

    # === Position modality - DETERMINISTIC for clean waypoint gradient ===
    # Previously we used uncertainty here for epistemic value, but this caused problems:
    # - Uncertainty dilutes expected C_pos preferences, penalizing exploration
    # - Agent prefers staying at certain positions over uncertain ones
    #
    # Now we use deterministic position observations so C_pos gradient works correctly.
    # Epistemic value comes from TILE uncertainty (landmark/arousal modalities) instead.
    A_pos_base = np.zeros((N_loc, N_loc, num_tile_states))

    for loc_idx in range(N_loc):
        for tile_state in range(num_tile_states):
            # Deterministic: position observation = true position
            # Small noise for numerical stability
            A_pos_base[loc_idx, loc_idx, tile_state] = 0.98
            # Tiny uniform spread to other positions
            noise = 0.02 / max(N_loc - 1, 1)
            for other_idx in range(N_loc):
                if other_idx != loc_idx:
                    A_pos_base[other_idx, loc_idx, tile_state] = noise

    # Normalize
    A_land_base /= A_land_base.sum(axis=0, keepdims=True)
    A_arousal_base /= A_arousal_base.sum(axis=0, keepdims=True)
    A_pos_base /= A_pos_base.sum(axis=0, keepdims=True)

    # Add batch dimension
    return [A_land_base[None, ...], A_arousal_base[None, ...], A_pos_base[None, ...]]


def build_C_with_waypoint(env: NicheGridEnv, planner, waypoint_strength: float = 2.0) -> List[np.ndarray]:
    """
    Build C matrix with preferences toward current waypoint.

    This gives the agent a gradient to follow toward the goal,
    even with a short planning horizon.

    Args:
        env: Environment
        planner: HierarchicalPlanner with current waypoint
        waypoint_strength: How strongly to prefer waypoint direction (0 = neutral)
    """
    num_cells = env.num_cells

    # Landmark: strong preference for goal, minimal preference for marked (avoid comfort trap)
    C_land = np.array([[0.0, 0.1, 8.0]])  # Reduced MarkedCue preference from 0.5 to 0.1

    # Arousal: mild preference for low arousal (reduced to avoid comfort trap)
    C_arousal = np.array([[0.5, 0.0]])  # Reduced from 2.0 to 0.5

    # Position: NOW we add preferences based on waypoint direction
    wp_x, wp_y = planner.get_current_waypoint()

    C_pos = np.zeros((1, num_cells))
    for loc_idx in range(num_cells):
        x, y = env._idx_to_xy(loc_idx)

        # Distance to waypoint
        dist = abs(x - wp_x) + abs(y - wp_y)

        # Preference: closer to waypoint = higher value
        # Use negative distance scaled by strength
        C_pos[0, loc_idx] = -dist * waypoint_strength

    # Shift so max is 0 (all values non-positive, closest to waypoint = 0)
    C_pos = C_pos - C_pos.min()

    return [C_land, C_arousal, C_pos]


def update_generative_model(
    env: NicheGridEnv,
    world_map,
    planner,
    builder_generation: bool = True,
    salience: float = 1.0,
    waypoint_strength: float = 2.0,
) -> Dict[str, List[np.ndarray]]:
    """
    Build a generative model that incorporates:
    1. Learned world knowledge (from world_map)
    2. Hierarchical goal structure (from planner)

    This should be called periodically (e.g., every N steps or when beliefs change significantly)
    to update the agent's model based on what it has learned.
    """
    A = build_A_with_exploration(env, world_map, salience=salience)
    B = build_B(env)
    C = build_C_with_waypoint(env, planner, waypoint_strength=waypoint_strength)
    D = build_D(env, builder_generation)

    return {"A": A, "B": B, "C": C, "D": D}
