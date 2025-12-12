# src/material_niches/world_map.py
"""
Persistent world map for active inference agent.

The agent learns about its environment over time:
- Unexplored cells: high uncertainty (epistemic value for visiting)
- Explored cells: low uncertainty (no need to revisit)

This creates natural exploration → exploitation behavior.
"""

import numpy as np
from typing import Dict, Tuple, Set, Optional


class WorldMap:
    """
    Tracks what the agent has learned about the world.

    Key insight: epistemic value should come from ACTUAL uncertainty,
    not artificial noise. The agent should:
    1. Start uncertain about what's at each location
    2. Learn by visiting locations
    3. Stop being curious about places it's already explored
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.num_cells = width * height

        # Track which cells have been visited
        self.visited: Set[int] = set()

        # Track observed tile types at each cell
        # None = unknown, 0 = WILD, 1 = MARKED
        self.observed_tiles: Dict[int, Optional[int]] = {
            i: None for i in range(self.num_cells)
        }

        # Confidence in observations (could use Dirichlet, but simple counts work)
        # Higher = more certain about what's at this location
        self.visit_counts: Dict[int, int] = {
            i: 0 for i in range(self.num_cells)
        }

    def _xy_to_idx(self, x: int, y: int) -> int:
        return y * self.width + x

    def _idx_to_xy(self, idx: int) -> Tuple[int, int]:
        y = idx // self.width
        x = idx % self.width
        return x, y

    def update(self, location_idx: int, observed_tile: int) -> None:
        """Update map with new observation."""
        self.visited.add(location_idx)
        self.observed_tiles[location_idx] = observed_tile
        self.visit_counts[location_idx] += 1

    def get_uncertainty(self, location_idx: int) -> float:
        """
        Get uncertainty level for a location.

        Returns value in [0, 1]:
        - 1.0 = completely unknown (never visited)
        - 0.0 = well known (visited multiple times)
        """
        visits = self.visit_counts[location_idx]
        if visits == 0:
            return 1.0
        else:
            # Decay uncertainty with visits
            # After 1 visit: 0.3, after 2: 0.15, after 3: 0.075, etc.
            return 0.3 ** visits

    def get_exploration_value(self) -> np.ndarray:
        """
        Get exploration value for each cell.

        Returns array of shape (num_cells,) with values in [0, 1].
        Higher values = more valuable to explore.
        """
        values = np.array([
            self.get_uncertainty(i) for i in range(self.num_cells)
        ])
        return values

    def get_frontier(self) -> Set[int]:
        """
        Get the 'frontier' - unexplored cells adjacent to explored ones.
        These are the most valuable cells to visit next.
        """
        frontier = set()
        for visited_idx in self.visited:
            x, y = self._idx_to_xy(visited_idx)
            # Check neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbor_idx = self._xy_to_idx(nx, ny)
                    if neighbor_idx not in self.visited:
                        frontier.add(neighbor_idx)
        return frontier

    def reset(self) -> None:
        """Reset the map (start of new learning episode)."""
        self.visited.clear()
        self.observed_tiles = {i: None for i in range(self.num_cells)}
        self.visit_counts = {i: 0 for i in range(self.num_cells)}

    def get_known_tile_map(self) -> np.ndarray:
        """
        Get the agent's current belief about tile states.

        Returns array of shape (num_cells,):
        - -1 = unknown
        - 0 = believed to be WILD
        - 1 = believed to be MARKED
        """
        tiles = np.full(self.num_cells, -1)
        for idx, tile in self.observed_tiles.items():
            if tile is not None:
                tiles[idx] = tile
        return tiles


class HierarchicalPlanner:
    """
    Two-level planning system:

    Level 1 (Strategic): Select which REGION/WAYPOINT to pursue
    Level 0 (Tactical): Plan short-term actions toward current subgoal

    This allows long-horizon goals with short planning windows.
    """

    def __init__(self, width: int, height: int,
                 goal_positions: list,
                 start_positions: list,
                 num_waypoints: int = 4):
        self.width = width
        self.height = height
        self.num_cells = width * height

        # Parse goal and start
        self.goal_y, self.goal_x = goal_positions[0]
        self.start_y, self.start_x = start_positions[0]

        # Generate waypoints from start to goal
        self.waypoints = self._generate_waypoints(num_waypoints)
        self.current_waypoint_idx = 0

    def _generate_waypoints(self, num_waypoints: int) -> list:
        """Generate waypoints along the path from start to goal."""
        waypoints = []

        for i in range(num_waypoints + 1):
            # Linear interpolation from start to goal
            t = i / num_waypoints
            x = int(self.start_x + t * (self.goal_x - self.start_x))
            y = int(self.start_y + t * (self.goal_y - self.start_y))
            waypoints.append((x, y))

        # Always include the actual goal
        waypoints.append((self.goal_x, self.goal_y))

        return waypoints

    def get_current_waypoint(self) -> Tuple[int, int]:
        """Get the current waypoint (x, y)."""
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return self.waypoints[-1]  # Return goal if we've passed all waypoints

    def update(self, agent_x: int, agent_y: int) -> bool:
        """
        Update waypoint progress based on agent position.

        Returns True if a waypoint was reached.
        """
        if self.current_waypoint_idx >= len(self.waypoints):
            return False

        wp_x, wp_y = self.waypoints[self.current_waypoint_idx]

        # Check if agent is close enough to current waypoint
        distance = abs(agent_x - wp_x) + abs(agent_y - wp_y)

        if distance <= 1:  # Within 1 step of waypoint
            self.current_waypoint_idx += 1
            return True

        return False

    def get_subgoal_preferences(self, current_x: int, current_y: int) -> np.ndarray:
        """
        Generate C-matrix preferences that guide agent toward current waypoint.

        Returns array of shape (num_cells,) with preference values.
        Higher = more preferred location.
        """
        preferences = np.zeros(self.num_cells)

        wp_x, wp_y = self.get_current_waypoint()

        for idx in range(self.num_cells):
            y = idx // self.width
            x = idx % self.width

            # Distance to waypoint (Manhattan)
            dist_to_waypoint = abs(x - wp_x) + abs(y - wp_y)

            # Preference decreases with distance (use negative because closer = better)
            # Scale so nearby cells are strongly preferred
            preferences[idx] = -dist_to_waypoint * 0.5

        # Normalize to reasonable range
        preferences = preferences - preferences.min()

        return preferences

    def reset(self) -> None:
        """Reset to first waypoint."""
        self.current_waypoint_idx = 0
