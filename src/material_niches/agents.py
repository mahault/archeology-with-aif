from typing import Dict
from pymdp.agent import Agent

def make_agent(gm: Dict, policy_len: int = 4) -> Agent:
    """
    Construct a pymdp.Agent given A,B,C,D and a short planning horizon.

    Model structure:
    - Factor 0: location (controlled by movement actions)
    - Factor 1: tile type (controlled by modify action)

    Both factors share the same action space (6 actions), but the agent
    only needs to select a single action that affects both.
    """
    agent = Agent(
        A=gm["A"],
        B=gm["B"],
        C=gm["C"],
        D=gm["D"],
        policy_len=policy_len,
        # Only factor 0 (location) is controllable - tile changes as side effect
        control_fac_idx=[0],
        # Use stochastic action selection for exploration
        action_selection="stochastic",
        # Temperature for softmax (higher = more random)
        alpha=4.0,
    )
    return agent
