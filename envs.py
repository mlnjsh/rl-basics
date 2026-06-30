"""envs.py - The Maze environment used throughout the RL course.

This module contains the `Maze` class that every notebook (except the
`MDP_introduction` notebook, which builds it from scratch for teaching) imports
with a single line:

    from envs import Maze

`Maze` is a small 5x5 grid-world written against the classic OpenAI Gym API
(`reset`, `step`, `render`). The agent starts in one corner and must reach the
goal in the opposite corner while respecting interior walls. It is deliberately
tiny so that tabular value/Q tables fit in a 5x5 array and every algorithm in
the course can be visualised on screen.
"""

from typing import Tuple, Dict, Optional, Iterable  # Type hints make the method signatures self-documenting.

import numpy as np                       # Numerical arrays (the wall map, distance map, rendered frame).
import gym                               # We subclass gym.Env so the env speaks the standard RL interface.
from gym import spaces                   # `spaces` describes the shape of actions and observations.

import pygame                            # Pygame is only used to *draw* the maze into an image.
from pygame import gfxdraw              # gfxdraw gives us anti-aliased polygons/circles for a clean picture.

import warnings
# Pygame emits a harmless UserWarning on some systems; silence it so the notebooks stay tidy.
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")


class Maze(gym.Env):
    """A 5x5 grid-world. State = (row, col). Actions = {0:UP, 1:RIGHT, 2:DOWN, 3:LEFT}."""

    def __init__(self, exploring_starts: bool = False,
                 shaped_rewards: bool = False, size: int = 5) -> None:
        super().__init__()
        # If True, reset() drops the agent at a random (non-goal) square -> helps Monte-Carlo "exploring starts".
        self.exploring_starts = exploring_starts
        # If True, reward is a smooth function of distance-to-goal instead of a flat -1 per step.
        self.shaped_rewards = shaped_rewards
        self.state = (size - 1, size - 1)            # Current agent position (placeholder; reset() sets the real start).
        self.goal = (size - 1, size - 1)             # The goal sits in the bottom-right corner.
        self.maze = self._create_maze(size=size)     # Dict: square -> list of squares you can legally move to.
        self.distances = self._compute_distances(self.goal, self.maze)  # Shortest-path distance of every square to the goal.

        self.action_space = spaces.Discrete(n=4)     # Four discrete actions.
        self.action_space.action_meanings = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: "LEFT"}  # Human-readable labels.
        self.observation_space = spaces.MultiDiscrete([size, size])     # A state is two integers in [0, size).

        self.screen = None                           # Lazily created pygame surface (only when we render).
        self.agent_transform = None

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """Apply `action`, return (next_state, reward, done, info) - the standard Gym tuple."""
        reward = self.compute_reward(self.state, action)   # Reward depends on where we are BEFORE moving.
        self.state = self._get_next_state(self.state, action)  # Move (or stay put if a wall blocks us).
        done = self.state == self.goal                     # The episode ends once we reach the goal.
        info = {}
        return self.state, reward, done, info

    def reset(self) -> Tuple[int, int]:
        """Start a new episode and return the initial state."""
        if self.exploring_starts:
            # Keep sampling until we get a square that is NOT the goal (otherwise the episode is already over).
            while self.state == self.goal:
                self.state = tuple(self.observation_space.sample())
        else:
            self.state = (0, 0)                           # Default fixed start: top-left corner.
        return self.state

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Draw the maze and return it as an RGB image array (used by the video/plot helpers)."""
        assert mode in ['human', 'rgb_array']

        screen_size = 600
        scale = screen_size / 5                            # Pixels per grid square.

        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface((screen_size, screen_size))

        surf = pygame.Surface((screen_size, screen_size))
        surf.fill((22, 36, 71))                            # Dark navy background.

        # Draw every wall: for each square, any neighbour it is NOT connected to gets a white border drawn.
        for row in range(5):
            for col in range(5):

                state = (row, col)
                for next_state in [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]:
                    if next_state not in self.maze[state]:

                        # Compute the rectangle that represents the wall between `state` and `next_state`.
                        row_diff, col_diff = np.subtract(next_state, state)
                        left = (col + (col_diff > 0)) * scale - 2 * (col_diff != 0)
                        right = ((col + 1) - (col_diff < 0)) * scale + 2 * (col_diff != 0)
                        top = (5 - (row + (row_diff > 0))) * scale - 2 * (row_diff != 0)
                        bottom = (5 - ((row + 1) - (row_diff < 0))) * scale + 2 * (row_diff != 0)

                        gfxdraw.filled_polygon(surf, [(left, bottom), (left, top), (right, top), (right, bottom)], (255, 255, 255))

        # Draw the goal square as a teal/green block.
        left, right, top, bottom = scale * 4 + 10, scale * 5 - 10, scale - 10, 10
        gfxdraw.filled_polygon(surf, [(left, bottom), (left, top), (right, top), (right, bottom)], (40, 199, 172))

        # Draw the agent as a FUCHSIA circle (RGB 255,0,255) at its current square.
        agent_row = int(screen_size - scale * (self.state[0] + .5))
        agent_col = int(scale * (self.state[1] + .5))
        gfxdraw.filled_circle(surf, agent_col, agent_row, int(scale * .6 / 2), (255, 0, 255))

        # Pygame's y-axis points down; flip so (0,0) is the bottom-left as humans expect.
        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        # Return the surface as an (H, W, 3) RGB array.
        return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        """Release the pygame window/surface."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def compute_reward(self, state: Tuple[int, int], action: int) -> float:
        """Reward for taking `action` in `state`.

        Default (sparse): -1 on every step until the goal -> encourages the SHORTEST path.
        Shaped: a smooth -distance/maxdistance signal -> easier for some algorithms to learn.
        """
        next_state = self._get_next_state(state, action)
        if self.shaped_rewards:
            return - (self.distances[next_state] / self.distances.max())
        return - float(state != self.goal)

    def simulate_step(self, state: Tuple[int, int], action: int):
        """Like step(), but for a hypothetical (state, action) WITHOUT changing the real env state.

        Dynamic-programming methods (policy/value iteration) use this to 'look ahead'.
        """
        reward = self.compute_reward(state, action)
        next_state = self._get_next_state(state, action)
        done = next_state == self.goal
        info = {}
        return next_state, reward, done, info

    def _get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Where does `action` take us from `state`? If a wall blocks the move, we stay put."""
        if action == 0:                                  # UP -> row decreases.
            next_state = (state[0] - 1, state[1])
        elif action == 1:                                # RIGHT -> column increases.
            next_state = (state[0], state[1] + 1)
        elif action == 2:                                # DOWN -> row increases.
            next_state = (state[0] + 1, state[1])
        elif action == 3:                                # LEFT -> column decreases.
            next_state = (state[0], state[1] - 1)
        else:
            raise ValueError("Action value not supported:", action)
        # The move is only allowed if `next_state` is a connected neighbour of `state`.
        if next_state in self.maze[state]:
            return next_state
        return state                                     # Blocked by a wall -> no movement.

    @staticmethod
    def _create_maze(size: int) -> Dict[Tuple[int, int], Iterable[Tuple[int, int]]]:
        """Build the connectivity map: start fully connected, then remove the outer border and interior walls."""
        # Every square initially lists its 4 orthogonal neighbours.
        maze = {(row, col): [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
                for row in range(size) for col in range(size)}

        # The four outer borders (so the agent cannot walk off the grid).
        left_edges = [[(row, 0), (row, -1)] for row in range(size)]
        right_edges = [[(row, size - 1), (row, size)] for row in range(size)]
        upper_edges = [[(0, col), (-1, col)] for col in range(size)]
        lower_edges = [[(size - 1, col), (size, col)] for col in range(size)]
        # The fixed interior walls that make this a maze (each pair = a blocked passage).
        walls = [
            [(1, 0), (1, 1)], [(2, 0), (2, 1)], [(3, 0), (3, 1)],
            [(1, 1), (1, 2)], [(2, 1), (2, 2)], [(3, 1), (3, 2)],
            [(3, 1), (4, 1)], [(0, 2), (1, 2)], [(1, 2), (1, 3)],
            [(2, 2), (3, 2)], [(2, 3), (3, 3)], [(2, 4), (3, 4)],
            [(4, 2), (4, 3)], [(1, 3), (1, 4)], [(2, 3), (2, 4)],
        ]

        obstacles = upper_edges + lower_edges + left_edges + right_edges + walls

        # Remove each blocked passage from both squares' neighbour lists.
        for src, dst in obstacles:
            maze[src].remove(dst)

            if dst in maze:
                maze[dst].remove(src)

        return maze

    @staticmethod
    def _compute_distances(goal: Tuple[int, int],
                           maze: Dict[Tuple[int, int], Iterable[Tuple[int, int]]]) -> np.ndarray:
        """Shortest-path distance from every square to the goal (a Dijkstra/BFS-style flood fill).

        Used only when `shaped_rewards=True` to turn distance into a reward signal.
        """
        distances = np.full((5, 5), np.inf)
        visited = set()
        distances[goal] = 0.

        while visited != set(maze):
            # Pick the unvisited square that is currently closest to the goal.
            sorted_dst = [(v // 5, v % 5) for v in distances.argsort(axis=None)]
            closest = next(x for x in sorted_dst if x not in visited)
            visited.add(closest)

            # Relax: a neighbour is at most one step further than the square we just settled.
            for neighbour in maze[closest]:
                distances[neighbour] = min(distances[neighbour], distances[closest] + 1)
        return distances
