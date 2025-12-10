# src/material_niches/agents.py

from typing import Dict
from pymdp.agent import Agent

def make_agent(gm: Dict, horizon: int = 10) -> Agent:
    """
    Construct a pymdp.Agent given A,B,C,D and planning horizon.
    """
    agent = Agent(
        A=gm["A"],
        B=gm["B"],
        C=gm["C"],
        D=gm["D"],
        policy_len=horizon,
        use_utility=False,      # we are using C as log-preferences directly
        inference_algo="VMP",   # or "MMP" depending on your taste
    )
    return agent
