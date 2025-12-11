# src/material_niches/simulate.py
from __future__ import annotations

import logging
from typing import Dict, List, Tuple
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from pymdp.agent import Agent

from .env import NicheGridEnv, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY, ACTION_MODIFY
from .generative_model import make_generative_model
from .agents import make_agent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    o_t = [jnp.array([[obs["landmark"]]]), jnp.array([[obs["arousal"]]])]

    # Initial state inference using D as empirical prior (no previous action)
    qs = agent.infer_states(observations=o_t, empirical_prior=agent.D)

    loc_traj = []
    tile_traj = []
    act_traj = []
    fe_traj = []

    # Track previous beliefs for empirical prior update
    qs_prev = qs

    for t in range(T):
        rng_key, subkey = jr.split(rng_key)

        # Infer policies - returns (q_pi, G) where G is negative expected free energy
        q_pi, neg_efe = agent.infer_policies(qs)

        # Sample action from policy distribution
        # rng_key needs shape (batch_size, 2) for vmap
        rng_key, action_key = jr.split(rng_key)
        action_keys = jnp.expand_dims(action_key, axis=0)  # shape (1, 2)
        action = agent.sample_action(q_pi, rng_key=action_keys)

        # action shape is (batch_size, num_factors) - extract scalar action
        # For single control factor, take first element
        action_int = int(action[0, 0])

        # Environment step
        state, obs = env.step(action_int)
        o_t = [jnp.array([[obs["landmark"]]]), jnp.array([[obs["arousal"]]])]

        # Update empirical prior based on action and previous beliefs
        empirical_prior, _ = agent.update_empirical_prior(action, qs_prev)

        # Infer new beliefs
        qs_prev = qs
        qs = agent.infer_states(observations=o_t, empirical_prior=empirical_prior)

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

    return {
        "loc": np.array(loc_traj),
        "tile": np.array(tile_traj),
        "action": np.array(act_traj),
        "free_energy": np.array(fe_traj),
    }


def run_generation(config,
                   builder_generation: bool,
                   n_episodes: int = 100,
                   T: int = 30,
                   seed: int | None = None):
    logger.info(f"Starting generation: builder={builder_generation}, n_episodes={n_episodes}, T={T}")
    rng = np.random.default_rng(seed)
    env = NicheGridEnv(config, rng=rng)
    logger.info(f"Environment created: {config.width}x{config.height} grid")

    gm = make_generative_model(env, builder_generation=builder_generation)
    logger.info("Generative model created")

    # Use shorter policy_len=2 for faster computation (36 policies vs 1296)
    agent = make_agent(gm, policy_len=2)
    logger.info("Agent created with policy_len=2")

    # JAX random key
    rng_key = jr.PRNGKey(seed if seed is not None else 0)

    all_trajs = []
    for ep in range(n_episodes):
        if ep % 5 == 0:
            logger.info(f"Running episode {ep+1}/{n_episodes}")
        rng_key, ep_key = jr.split(rng_key)
        traj = run_episode(env, agent, T=T, episode_num=ep, rng_key=ep_key)
        all_trajs.append(traj)

    logger.info(f"Generation complete: {n_episodes} episodes finished")
    return env, all_trajs
