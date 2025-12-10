# src/material_niches/simulate.py

from typing import Dict, List, Tuple
import numpy as np
from pymdp.agent import Agent

from .env import NicheGridEnv
from .generative_model import make_generative_model
from .agents import make_agent

def run_episode(env: NicheGridEnv,
                agent: Agent,
                T: int = 30) -> Dict[str, np.ndarray]:
    """
    Run a single episode with a pymdp.Agent in the niche-construction grid.

    Returns a dict of trajectories: states, actions, observations, free energy.
    """
    state = env.reset()
    obs = env._render_observation(state)

    # In pymdp, observations are indices per modality
    o_t = [obs["landmark"], obs["arousal"]]
    agent.reset()
    agent.infer_states(o_t)

    loc_traj = []
    tile_traj = []
    act_traj = []
    fe_traj = []

    for t in range(T):
        # infer policies & sample action
        q_pi, neg_efe = agent.infer_policies()
        action = agent.sample_action()

        # environment step
        state, obs = env.step(action)
        o_t = [obs["landmark"], obs["arousal"]]

        qs = agent.infer_states(o_t)

        # logging
        loc_traj.append(state["loc"])
        tile_traj.append(state["tile"])
        act_traj.append(action)
        fe_traj.append(-neg_efe.max())  # crude FE summary per timestep

    return {
        "loc": np.array(loc_traj),
        "tile": np.array(tile_traj),
        "action": np.array(act_traj),
        "free_energy": np.array(fe_traj),
    }

def run_generation(config, builder_generation: bool,
                   n_episodes: int = 100,
                   T: int = 30,
                   seed: int | None = None):
    """
    Helper to run many episodes for one generation (builders or inheritors).
    """
    rng = np.random.default_rng(seed)
    env = NicheGridEnv(config, rng=rng)
    gm = make_generative_model(env, builder_generation=builder_generation)
    agent = make_agent(gm)

    all_trajs = []
    for ep in range(n_episodes):
        traj = run_episode(env, agent, T=T)
        all_trajs.append(traj)

    return env, all_trajs
