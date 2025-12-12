from typing import Dict, Literal
from pymdp.agent import Agent

def make_agent(
    gm: Dict,
    policy_len: int = 4,
    action_selection: Literal["stochastic", "deterministic"] = "stochastic",
    alpha: float = 8.0,
) -> Agent:
    """
    Construct a pymdp.Agent given A,B,C,D and a short planning horizon.

    Model structure:
    - Factor 0: location (controlled by movement actions)
    - Factor 1: tile type (changes as side effect of same action)

    Modalities:
    - Modality 0 (landmark): depends on location and tile type [0, 1]
    - Modality 1 (arousal): depends on location and tile type [0, 1]
    - Modality 2 (position): depends only on location [0]

    Both factors are affected by the same action (shared control).
    This reduces policy space from 6^(2*policy_len) to 6^policy_len.
    """
    agent = Agent(
        A=gm["A"],
        B=gm["B"],
        C=gm["C"],
        D=gm["D"],
        policy_len=policy_len,
        # Single action controls both factors
        num_controls=[6],
        # Both B factors depend on the same action (action 0)
        B_action_dependencies=[[0], [0]],
        # A dependencies: which hidden state factors each modality depends on
        # Modality 0 (landmark): depends on location (0) and tile (1)
        # Modality 1 (arousal): depends on location (0) and tile (1)
        # Modality 2 (position): depends on location (0) and tile (1)
        # NOTE: All A matrices have shape (batch, n_obs, n_loc, n_tile) so all depend on [0,1]
        A_dependencies=[[0, 1], [0, 1], [0, 1]],
        # Enable epistemic value (information gain about states)
        use_states_info_gain=True,
        # Enable utility (preference satisfaction)
        use_utility=True,
        # Action selection mode: "stochastic" (softmax) or "deterministic" (greedy)
        action_selection=action_selection,
        # Temperature for softmax (only used if stochastic)
        # Higher alpha = more greedy, lower = more random
        alpha=alpha,
    )
    return agent
