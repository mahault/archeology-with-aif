# src/material_niches/simulate.py
from __future__ import annotations

import logging
from typing import Dict, List, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from pymdp.agent import Agent

from .env import NicheGridEnv, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY, ACTION_MODIFY
from .generative_model import make_generative_model, update_generative_model
from .agents import make_agent
from .world_map import WorldMap, HierarchicalPlanner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# JIT-compiled agent functions for speed
@jax.jit
def _jit_infer_and_act(agent, qs, rng_key):
    """JIT-compiled policy inference and action sampling."""
    q_pi, neg_efe = agent.infer_policies(qs)
    action = agent.sample_action(q_pi, rng_key=rng_key)
    return action, q_pi, neg_efe


@jax.jit
def _jit_infer_states(agent, observations, empirical_prior):
    """JIT-compiled state inference."""
    return agent.infer_states(observations=observations, empirical_prior=empirical_prior)


@jax.jit
def _jit_update_prior(agent, action, qs_prev):
    """JIT-compiled empirical prior update."""
    return agent.update_empirical_prior(action, qs_prev)

ACTION_NAMES = {
    ACTION_UP: "UP",
    ACTION_DOWN: "DOWN",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
    ACTION_STAY: "STAY",
    ACTION_MODIFY: "MODIFY"
}


def render_grid(env: NicheGridEnv, agent_loc: int = None) -> str:
    """Render a text visualization of the grid."""
    lines = []
    for y in range(env.config.height):
        row = []
        for x in range(env.config.width):
            idx = env._xy_to_idx(x, y)
            is_goal = (y, x) in env.config.goal_positions
            is_start = (y, x) in env.config.start_positions
            tile = env.tile_map[idx]

            if agent_loc is not None and idx == agent_loc:
                char = "A"  # Agent
            elif is_goal:
                char = "G"  # Goal
            elif is_start:
                char = "S"  # Start
            elif tile == env.MARKED:
                char = "#"  # Marked/path
            else:
                char = "."  # Wild
            row.append(char)
        lines.append(" ".join(row))
    return "\n".join(lines)


def run_episode(env: NicheGridEnv,
                agent: Agent,
                T: int = 30,
                episode_num: int = 0,
                rng_key=None,
                verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    Run a single episode with a pymdp.Agent in the niche-construction grid.

    Returns a dict of trajectories: states, actions, observations, free energy.
    """
    if rng_key is None:
        rng_key = jr.PRNGKey(episode_num)

    state = env.reset()
    obs = env._render_observation(state)

    x, y = env._idx_to_xy(state['loc'])
    if verbose:
        logger.info(f"Episode {episode_num}: Starting at (x={x}, y={y})")
        logger.info(f"Initial grid:\n{render_grid(env, state['loc'])}")

    # In JAX pymdp, observations need batch dimension: (batch_size, ...)
    # Our batch_size is 1
    # Three modalities: landmark, arousal, position
    o_t = [
        jnp.array([[obs["landmark"]]]),
        jnp.array([[obs["arousal"]]]),
        jnp.array([[obs["position"]]])
    ]

    # Initial state inference using D as empirical prior (no previous action)
    # Use JIT-compiled function
    qs = _jit_infer_states(agent, o_t, agent.D)

    loc_traj = []
    tile_traj = []
    act_traj = []
    fe_traj = []

    # Track previous beliefs for empirical prior update
    qs_prev = qs

    for t in range(T):
        # Sample action using JIT-compiled function
        # rng_key needs shape (batch_size, 2) for vmap
        rng_key, action_key = jr.split(rng_key)
        action_keys = jnp.expand_dims(action_key, axis=0)  # shape (1, 2)

        # JIT-compiled policy inference and action sampling
        action, q_pi, neg_efe = _jit_infer_and_act(agent, qs, action_keys)

        # Debug: log epistemic value diagnostics
        if verbose and t == 0:
            neg_efe_np = np.array(neg_efe)
            q_pi_np = np.array(q_pi)
            logger.info(f"  [DEBUG] neg_efe range: {neg_efe_np.min():.3f} to {neg_efe_np.max():.3f}, spread: {neg_efe_np.max() - neg_efe_np.min():.3f}")
            logger.info(f"  [DEBUG] q_pi entropy: {-(q_pi_np * np.log(q_pi_np + 1e-10)).sum():.3f}")
            logger.info(f"  [DEBUG] num policies: {len(neg_efe_np.flatten())}, top 5 neg_efe: {np.sort(neg_efe_np.flatten())[-5:]}")

        # action shape is (batch_size, num_factors) - extract scalar action
        # For single control factor, take first element
        action_int = int(action[0, 0])

        # Environment step (Python, not JIT-able)
        state, obs = env.step(action_int)
        o_t = [
            jnp.array([[obs["landmark"]]]),
            jnp.array([[obs["arousal"]]]),
            jnp.array([[obs["position"]]])
        ]

        # Update empirical prior based on action and previous beliefs (JIT-compiled)
        empirical_prior, _ = _jit_update_prior(agent, action, qs_prev)

        # Infer new beliefs (JIT-compiled)
        qs_prev = qs
        qs = _jit_infer_states(agent, o_t, empirical_prior)

        # Log trajectory
        loc_traj.append(state["loc"])
        tile_traj.append(state["tile"])
        act_traj.append(action_int)

        # neg_efe is expected free energy (higher = better), use max as FE summary
        fe_val = float(neg_efe.max()) if neg_efe.size > 0 else 0.0
        fe_traj.append(-fe_val)  # convert to free energy (lower = better)

        if verbose and t % 5 == 0:
            x, y = env._idx_to_xy(state['loc'])
            logger.info(f"  t={t}: pos=({x},{y}), action={ACTION_NAMES.get(action_int, action_int)}, tile={state['tile']}, FE={-fe_val:.3f}")

    if verbose:
        x, y = env._idx_to_xy(state['loc'])
        logger.info(f"Episode {episode_num} ended at (x={x}, y={y})")
        logger.info(f"Final grid:\n{render_grid(env, state['loc'])}")
        # Count marked tiles
        marked_count = np.sum(env.tile_map == env.MARKED)
        logger.info(f"Marked tiles: {marked_count}/{env.num_cells}")

    return {
        "loc": np.array(loc_traj),
        "tile": np.array(tile_traj),
        "action": np.array(act_traj),
        "free_energy": np.array(fe_traj),
    }


def run_episode_with_learning(
    env: NicheGridEnv,
    world_map: WorldMap,
    planner: HierarchicalPlanner,
    T: int = 30,
    episode_num: int = 0,
    builder_generation: bool = True,
    rng_key=None,
    verbose: bool = False,
    model_update_interval: int = 5,
    waypoint_strength: float = 2.0,
    action_selection: str = "stochastic",
) -> Dict[str, np.ndarray]:
    """
    Run a single episode WITH learning and hierarchical planning.

    Key differences from run_episode:
    1. Uses WorldMap to track what agent has learned (exploration state)
    2. Uses HierarchicalPlanner for waypoint-based navigation
    3. Periodically updates the generative model based on learned knowledge

    Args:
        env: The environment
        world_map: Persistent map of what agent has learned
        planner: Hierarchical planner with waypoints
        T: Episode length
        episode_num: For logging
        builder_generation: Whether this is builder generation
        rng_key: JAX random key
        verbose: Whether to log details
        model_update_interval: How often to rebuild the agent's model
        waypoint_strength: How strongly to prefer waypoint direction
    """
    if rng_key is None:
        rng_key = jr.PRNGKey(episode_num)

    state = env.reset()
    obs = env._render_observation(state)

    # Update world map with initial observation
    world_map.update(state['loc'], state['tile'])

    x, y = env._idx_to_xy(state['loc'])
    if verbose:
        logger.info(f"Episode {episode_num}: Starting at (x={x}, y={y})")
        wp_x, wp_y = planner.get_current_waypoint()
        logger.info(f"  Current waypoint: ({wp_x}, {wp_y}), waypoint_idx={planner.current_waypoint_idx}")
        logger.info(f"Initial grid:\n{render_grid(env, state['loc'])}")

    # Check if agent is already at first waypoint (common at start)
    agent_x, agent_y = env._idx_to_xy(state['loc'])
    if planner.update(agent_x, agent_y) and verbose:
        logger.info(f"  Starting at waypoint {planner.current_waypoint_idx - 1}, now targeting {planner.current_waypoint_idx}")

    # Build initial generative model with learning and planning
    gm = update_generative_model(
        env, world_map, planner,
        builder_generation=builder_generation,
        waypoint_strength=waypoint_strength
    )
    agent = make_agent(gm, policy_len=3, action_selection=action_selection)

    # Initial observation
    o_t = [
        jnp.array([[obs["landmark"]]]),
        jnp.array([[obs["arousal"]]]),
        jnp.array([[obs["position"]]])
    ]

    # Initial state inference
    qs = _jit_infer_states(agent, o_t, agent.D)

    loc_traj = []
    tile_traj = []
    act_traj = []
    fe_traj = []
    waypoint_reached = []

    qs_prev = qs

    for t in range(T):
        # Periodically update the model based on what we've learned
        if t > 0 and t % model_update_interval == 0:
            # Rebuild model with updated world knowledge and waypoint
            gm = update_generative_model(
                env, world_map, planner,
                builder_generation=builder_generation,
                waypoint_strength=waypoint_strength
            )
            agent = make_agent(gm, policy_len=3, action_selection=action_selection)
            # Re-infer states with new model
            qs = _jit_infer_states(agent, o_t, agent.D)
            qs_prev = qs

            if verbose:
                wp_x, wp_y = planner.get_current_waypoint()
                explored = len(world_map.visited)
                logger.info(f"  [MODEL UPDATE t={t}] waypoint=({wp_x},{wp_y}), explored={explored}/{env.num_cells}")

        # Sample action
        rng_key, action_key = jr.split(rng_key)
        action_keys = jnp.expand_dims(action_key, axis=0)

        action, q_pi, neg_efe = _jit_infer_and_act(agent, qs, action_keys)

        # Debug logging
        if verbose and t == 0:
            neg_efe_np = np.array(neg_efe)
            q_pi_np = np.array(q_pi)
            logger.info(f"  [DEBUG] neg_efe range: {neg_efe_np.min():.3f} to {neg_efe_np.max():.3f}, spread: {neg_efe_np.max() - neg_efe_np.min():.3f}")
            logger.info(f"  [DEBUG] q_pi entropy: {-(q_pi_np * np.log(q_pi_np + 1e-10)).sum():.3f}")

        action_int = int(action[0, 0])

        # Environment step
        state, obs = env.step(action_int)

        # Update world map with new observation
        world_map.update(state['loc'], state['tile'])

        # Update hierarchical planner (check if waypoint reached)
        agent_x, agent_y = env._idx_to_xy(state['loc'])
        reached = planner.update(agent_x, agent_y)
        if reached:
            if verbose:
                logger.info(f"  [WAYPOINT REACHED t={t}] Now targeting waypoint {planner.current_waypoint_idx}")
            # IMPORTANT: Immediately rebuild model with new waypoint to avoid stale preferences
            gm = update_generative_model(
                env, world_map, planner,
                builder_generation=builder_generation,
                waypoint_strength=waypoint_strength
            )
            agent = make_agent(gm, policy_len=3, action_selection=action_selection)

        o_t = [
            jnp.array([[obs["landmark"]]]),
            jnp.array([[obs["arousal"]]]),
            jnp.array([[obs["position"]]])
        ]

        # Update beliefs
        empirical_prior, _ = _jit_update_prior(agent, action, qs_prev)
        qs_prev = qs
        qs = _jit_infer_states(agent, o_t, empirical_prior)

        # Log trajectory
        loc_traj.append(state["loc"])
        tile_traj.append(state["tile"])
        act_traj.append(action_int)
        waypoint_reached.append(reached)

        fe_val = float(neg_efe.max()) if neg_efe.size > 0 else 0.0
        fe_traj.append(-fe_val)

        if verbose and t % 5 == 0:
            x, y = env._idx_to_xy(state['loc'])
            wp_x, wp_y = planner.get_current_waypoint()
            logger.info(f"  t={t}: pos=({x},{y}), action={ACTION_NAMES.get(action_int, action_int)}, tile={state['tile']}, FE={-fe_val:.3f}, wp=({wp_x},{wp_y})")

    if verbose:
        x, y = env._idx_to_xy(state['loc'])
        logger.info(f"Episode {episode_num} ended at (x={x}, y={y})")
        logger.info(f"Final grid:\n{render_grid(env, state['loc'])}")
        marked_count = np.sum(env.tile_map == env.MARKED)
        explored_count = len(world_map.visited)
        logger.info(f"Marked tiles: {marked_count}/{env.num_cells}, Explored: {explored_count}/{env.num_cells}")

        # Check if goal was reached
        goal_y, goal_x = env.config.goal_positions[0]
        goal_idx = env._xy_to_idx(goal_x, goal_y)
        if state['loc'] == goal_idx:
            logger.info(f"*** GOAL REACHED! ***")

    return {
        "loc": np.array(loc_traj),
        "tile": np.array(tile_traj),
        "action": np.array(act_traj),
        "free_energy": np.array(fe_traj),
        "waypoint_reached": np.array(waypoint_reached),
    }


def run_generation_with_learning(
    config,
    builder_generation: bool,
    n_episodes: int = 100,
    T: int = 30,
    seed: int | None = None,
    verbose_episodes: list = None,
    num_waypoints: int = 4,
    waypoint_strength: float = 2.0,
    reset_map_each_episode: bool = False,
    action_selection: str = "stochastic",
):
    """
    Run multiple episodes with learning and hierarchical planning.

    Args:
        config: Grid configuration
        builder_generation: Whether this is builder generation
        n_episodes: Number of episodes
        T: Episode length
        seed: Random seed
        verbose_episodes: Which episodes to log in detail
        num_waypoints: Number of waypoints between start and goal
        waypoint_strength: How strongly to prefer waypoint direction
        reset_map_each_episode: If True, agent forgets between episodes
    """
    if verbose_episodes is None:
        verbose_episodes = list(range(n_episodes))

    gen_type = "BUILDER" if builder_generation else "INHERITOR"
    logger.info(f"{'='*60}")
    logger.info(f"Starting {gen_type} generation WITH LEARNING: n_episodes={n_episodes}, T={T}")
    logger.info(f"  waypoints={num_waypoints}, waypoint_strength={waypoint_strength}")
    logger.info(f"{'='*60}")

    rng = np.random.default_rng(seed)
    env = NicheGridEnv(config, rng=rng)
    logger.info(f"Environment: {config.width}x{config.height} grid")
    logger.info(f"Start positions: {config.start_positions}")
    logger.info(f"Goal positions: {config.goal_positions}")
    logger.info(f"Initial grid:\n{render_grid(env)}")

    # Initialize world map (persistent across episodes unless reset)
    world_map = WorldMap(config.width, config.height)

    # Initialize hierarchical planner
    planner = HierarchicalPlanner(
        config.width, config.height,
        config.goal_positions, config.start_positions,
        num_waypoints=num_waypoints
    )
    logger.info(f"Waypoints: {planner.waypoints}")

    rng_key = jr.PRNGKey(seed if seed is not None else 0)

    all_trajs = []
    goals_reached = 0

    for ep in range(n_episodes):
        verbose = ep in verbose_episodes
        if ep % 5 == 0:
            logger.info(f"--- Episode {ep+1}/{n_episodes} ---")

        # Optionally reset map and planner each episode
        if reset_map_each_episode:
            world_map.reset()
        planner.reset()

        rng_key, ep_key = jr.split(rng_key)
        traj = run_episode_with_learning(
            env, world_map, planner,
            T=T,
            episode_num=ep,
            builder_generation=builder_generation,
            rng_key=ep_key,
            verbose=verbose,
            waypoint_strength=waypoint_strength,
            action_selection=action_selection,
        )
        all_trajs.append(traj)

        # Check if goal was reached
        goal_y, goal_x = config.goal_positions[0]
        goal_idx = env._xy_to_idx(goal_x, goal_y)
        final_loc = traj["loc"][-1]
        if final_loc == goal_idx:
            goals_reached += 1

    # Final summary
    marked_count = np.sum(env.tile_map == env.MARKED)
    explored_count = len(world_map.visited)
    logger.info(f"{'='*60}")
    logger.info(f"{gen_type} generation complete: {n_episodes} episodes")
    logger.info(f"Goals reached: {goals_reached}/{n_episodes}")
    logger.info(f"Final marked tiles: {marked_count}/{env.num_cells}")
    logger.info(f"Total explored: {explored_count}/{env.num_cells}")
    logger.info(f"Final grid:\n{render_grid(env)}")
    logger.info(f"{'='*60}")

    return env, all_trajs, world_map


def run_generation(config,
                   builder_generation: bool,
                   n_episodes: int = 100,
                   T: int = 30,
                   seed: int | None = None,
                   verbose_episodes: list = None):
    """
    Run multiple episodes of a generation.

    Args:
        verbose_episodes: List of episode numbers to log in detail (default: first and last)
    """
    if verbose_episodes is None:
        # Log every episode
        verbose_episodes = list(range(n_episodes))

    gen_type = "BUILDER" if builder_generation else "INHERITOR"
    logger.info(f"{'='*60}")
    logger.info(f"Starting {gen_type} generation: n_episodes={n_episodes}, T={T}")
    logger.info(f"{'='*60}")

    rng = np.random.default_rng(seed)
    env = NicheGridEnv(config, rng=rng)
    logger.info(f"Environment: {config.width}x{config.height} grid")
    logger.info(f"Start positions: {config.start_positions}")
    logger.info(f"Goal positions: {config.goal_positions}")
    logger.info(f"Initial grid:\n{render_grid(env)}")

    gm = make_generative_model(env, builder_generation=builder_generation)
    logger.info("Generative model created")

    # Use policy_len=3 for reasonable planning horizon (216 policies)
    # Longer horizons help the agent plan paths to the goal
    agent = make_agent(gm, policy_len=3)
    logger.info(f"Agent: policy_len=3, num_policies={agent.policies.shape[0]}, action_selection={agent.action_selection}")

    # JAX random key
    rng_key = jr.PRNGKey(seed if seed is not None else 0)

    all_trajs = []
    for ep in range(n_episodes):
        verbose = ep in verbose_episodes
        if ep % 5 == 0:
            logger.info(f"--- Episode {ep+1}/{n_episodes} ---")
        rng_key, ep_key = jr.split(rng_key)
        traj = run_episode(env, agent, T=T, episode_num=ep, rng_key=ep_key, verbose=verbose)
        all_trajs.append(traj)

    # Final summary
    marked_count = np.sum(env.tile_map == env.MARKED)
    logger.info(f"{'='*60}")
    logger.info(f"{gen_type} generation complete: {n_episodes} episodes")
    logger.info(f"Final marked tiles: {marked_count}/{env.num_cells}")
    logger.info(f"Final grid:\n{render_grid(env)}")
    logger.info(f"{'='*60}")

    return env, all_trajs
