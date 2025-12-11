# src/material_niches/env.py

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import numpy as np

Move = Tuple[int, int]

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STAY = 4
ACTION_MODIFY = 5

@dataclass
class GridConfig:
    width: int
    height: int
    start_positions: List[Tuple[int, int]]
    goal_positions: List[Tuple[int, int]]

class NicheGridEnv:
    """
    Simple gridworld generative process for 'Accumulation of Certainty'.

    - Grid cells: (y, x) indices
    - Two tile types: WILD (0), MARKED (1)
    - Agent has position and can MOVE / MODIFY tiles.
    - Environment stores the 'map' of which tiles are wild vs marked.
    """

    WILD = 0
    MARKED = 1

    def __init__(
        self,
        config: GridConfig,
        rng: Optional[np.random.Generator] = None,
        initial_tile_map: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.rng = rng or np.random.default_rng()

        self.num_cells = config.width * config.height

        if initial_tile_map is not None:
            assert initial_tile_map.shape == (self.num_cells,)
            self.tile_map = initial_tile_map.astype(int).copy()
        else:
            # default: all wild
            self.tile_map = np.full(self.num_cells, self.WILD, dtype=int)

            # mark goal cells as MARKED if desired
            for (y, x) in config.goal_positions:
                idx = self._xy_to_idx(x, y)
                self.tile_map[idx] = self.MARKED

        self.agent_idx: Optional[int] = None

    def _xy_to_idx(self, x: int, y: int) -> int:
        return y * self.config.width + x

    def _idx_to_xy(self, idx: int) -> Tuple[int, int]:
        y = idx // self.config.width
        x = idx % self.config.width
        return x, y

    def reset(self) -> Dict:
        """Reset environment to start of an episode."""
        start_y, start_x = self.config.start_positions[
            self.rng.integers(len(self.config.start_positions))
        ]
        self.agent_idx = self._xy_to_idx(start_x, start_y)
        return self._get_state()

    def _get_state(self) -> Dict:
        """Return current hidden state in factorized form."""
        tile_type = self.tile_map[self.agent_idx]
        return {
            "loc": self.agent_idx,
            "tile": int(tile_type),
        }

    def step(self, action: int) -> Tuple[Dict, Dict]:
        """Apply action, update environment, return (state, observation)."""

        if action in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY):
            self._move_agent(action)
        elif action == ACTION_MODIFY:
            # Niche construction: Wild -> Marked at current cell
            if self.tile_map[self.agent_idx] == self.WILD:
                self.tile_map[self.agent_idx] = self.MARKED

        state = self._get_state()
        obs = self._render_observation(state)
        return state, obs

    def _move_agent(self, action: int) -> None:
        x, y = self._idx_to_xy(self.agent_idx)
        if action == ACTION_UP:
            y = max(0, y - 1)
        elif action == ACTION_DOWN:
            y = min(self.config.height - 1, y + 1)
        elif action == ACTION_LEFT:
            x = max(0, x - 1)
        elif action == ACTION_RIGHT:
            x = min(self.config.width - 1, x + 1)
        # ACTION_STAY: do nothing

        self.agent_idx = self._xy_to_idx(x, y)

    def _render_observation(self, state: Dict) -> Dict:
        """
        Map hidden state to discrete observations.

        This is the generative PROCESS side; the corresponding A-matrix will
        be built to match this logic (Wild vs Marked vs Goal).
        """
        idx = state["loc"]
        tile_type = state["tile"]

        x, y = self._idx_to_xy(idx)
        is_goal = (y, x) in self.config.goal_positions

        # modality 1: landmark
        if is_goal:
            land_obs = 2  # GoalCue
        elif tile_type == self.MARKED:
            land_obs = 1  # MarkedCue
        else:
            land_obs = 0  # WildCue

        # modality 2: arousal (very simple heuristic)
        if tile_type == self.WILD and not is_goal:
            arousal_obs = 1  # High
        else:
            arousal_obs = 0  # Low

        return {
            "landmark": land_obs,
            "arousal": arousal_obs,
        }

    def remove_path_segment(self, cells_to_remove: List[Tuple[int, int]]) -> None:
        """
        'Taphonomy' operator: remove MARKED status from a segment of the path.
        """
        for (y, x) in cells_to_remove:
            idx = self._xy_to_idx(x, y)
            self.tile_map[idx] = self.WILD

    def clone_with_tilemap(self, tile_map: np.ndarray) -> "NicheGridEnv":
        """
        Convenience: make a new env with the same config but a supplied tile_map.
        """
        return NicheGridEnv(
            config=self.config,
            rng=self.rng,
            initial_tile_map=tile_map,
        )
