"""utils_frozenlake.py - Plotting and evaluation helpers for the Gymnasium RL course.

This is the Gymnasium (1.x) migration of the original utils.py that served the
Maze-based course. Function names and the module layout are kept the same, so a
notebook only needs to change its import line:

    from utils_frozenlake import plot_policy, plot_values, plot_action_values, test_agent

What changed relative to the old utils.py, and why:

  1. Gymnasium API everywhere.
       reset() returns (state, info); step() returns 5 values
       (state, reward, terminated, truncated, info); the render mode is fixed
       at gym.make(..., render_mode="rgb_array"), so render() takes no argument.

  2. Integer states, flexible grid size.
       FrozenLake states are single integers (0..15 on the 4x4 map). All tabular
       plot functions accept flat tables - V of shape (n_states,) and Q of shape
       (n_states, n_actions) - and reshape internally. The grid side is inferred
       as sqrt(n_states), so the SAME functions work unchanged on the 8x8 map.

  3. Terminal-tile masking.
       Pass `env` to the plotting functions and holes/gift are labelled H / G
       instead of showing a meaningless arrow or value. This needs the map
       layout, which lives in env.unwrapped.desc.

  4. FrozenLake action order.
       0=LEFT, 1=DOWN, 2=RIGHT, 3=UP (the old maze used 0=UP,1=RIGHT,2=DOWN,3=LEFT).
       All defaults below use the FrozenLake order.

`torch` is imported lazily (only the deep-RL helpers need it) so the tabular
notebooks can `import utils_frozenlake` on a machine without PyTorch.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation
from IPython.display import HTML

try:
    import torch
except ImportError:                      # torch is optional for the tabular notebooks
    torch = None

# FrozenLake action encoding, used as the default everywhere below.
FROZENLAKE_ACTIONS = {0: 'L', 1: 'D', 2: 'R', 3: 'U'}


# =============================================================================
# Internal helpers (not exported)
# =============================================================================
def _grid_side(n_states):
    """Infer the square grid side from the number of states (16 -> 4, 64 -> 8)."""
    side = int(round(np.sqrt(n_states)))
    if side * side != n_states:
        raise ValueError(f"{n_states} states do not form a square grid.")
    return side


def _to_grid_v(table):
    """Reshape a V table into (side, side) grid form.

    Accepts a flat (n_states,) vector or an already-square (side, side) grid.
    V and Q need separate converters because shape alone is ambiguous: a (4, 4)
    array could be a 4x4 V grid or a flat Q table of 4 states x 4 actions.
    Each plot function knows which kind it expects, so no guessing is needed.
    """
    table = np.asarray(table)
    if table.ndim == 1:                                   # (16,) -> (4, 4)
        side = _grid_side(table.shape[0])
        return table.reshape(side, side)
    if table.ndim == 2 and table.shape[0] == table.shape[1]:
        return table                                      # already a grid
    raise ValueError(f"Expected a V table of shape (n_states,) or (side, side), got {table.shape}.")


def _to_grid_q(table):
    """Reshape a Q table into (side, side, n_actions) grid form.

    Accepts a flat (n_states, n_actions) table or an already-shaped
    (side, side, n_actions) grid.
    """
    table = np.asarray(table)
    if table.ndim == 2:                                   # (16, 4) -> (4, 4, 4)
        side = _grid_side(table.shape[0])
        return table.reshape(side, side, table.shape[1])
    if table.ndim == 3:
        return table                                      # already a grid
    raise ValueError(f"Expected a Q table of shape (n_states, n_actions) or (side, side, n_actions), got {table.shape}.")


def _desc_letters(env, side):
    """Return the map letters (S/F/H/G) as a (side, side) array, or None if no env given."""
    if env is None:
        return None
    desc = np.array([[ch.decode() for ch in row] for row in env.unwrapped.desc])
    if desc.shape != (side, side):
        raise ValueError(f"env map is {desc.shape}, but the table implies {side}x{side}.")
    return desc


# =============================================================================
# Video / rollout helpers
# =============================================================================
def display_video(frames, fps=4):
    """Turn a list of RGB frames into an HTML5 video the notebook can play inline."""
    # Adapted from: https://colab.research.google.com/github/deepmind/dm_control/blob/master/tutorial.ipynb
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')                 # non-GUI backend: build frames without opening a window
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    matplotlib.use(orig_backend)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
        im.set_data(frame)
        return [im]
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=1000 // fps, blit=True, repeat=False)
    return HTML(anim.to_html5_video())


def test_agent(environment, policy, episodes=5):
    """Roll out a tabular `policy` for several episodes and return a playable video.

    `policy(state)` may return either a single action (greedy policy) or a
    probability vector over the actions (stochastic policy) - both are handled.
    The environment must have been created with render_mode="rgb_array".
    """
    n_actions = environment.action_space.n
    frames = []
    for _ in range(episodes):
        state, info = environment.reset()
        done = False
        frames.append(environment.render())
        while not done:
            p = policy(state)
            if isinstance(p, np.ndarray):
                action = np.random.choice(n_actions, p=p)   # stochastic: sample
            else:
                action = int(p)                             # deterministic: use directly
            state, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated
            frames.append(environment.render())
    return display_video(frames)


def test_env(environment, episodes=5):
    """Roll out a RANDOM agent - used in the intro to watch an untrained environment."""
    frames = []
    for _ in range(episodes):
        state, info = environment.reset()
        done = False
        frames.append(environment.render())
        while not done:
            action = environment.action_space.sample()
            state, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated
            frames.append(environment.render())
    return display_video(frames)


def test_policy_network(env, policy, episodes=10):
    """Roll out a neural-network policy (policy-gradient sections) and return a video.

    `policy` is a torch network mapping a state tensor to action probabilities.
    Works for both vector observations (CartPole) and integer observations
    (FrozenLake, one-hot encoded before the forward pass).
    """
    if torch is None:
        raise ImportError("test_policy_network requires PyTorch.")
    frames = []
    for _ in range(episodes):
        state, info = env.reset()
        done = False
        frames.append(env.render())
        while not done:
            if np.isscalar(state):        # integer state -> one-hot vector the network can eat
                x = torch.zeros(1, env.observation_space.n)
                x[0, state] = 1.0
            else:                         # vector state -> batched float tensor
                x = torch.from_numpy(np.asarray(state)).unsqueeze(0).float()
            action = policy(x).multinomial(1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frames.append(env.render())
    return display_video(frames)


def evaluate_policy(environment, policy, episodes=1000):
    """Return the average total reward of `policy` over many episodes (no video).

    On FrozenLake the total reward per episode is 0 or 1, so this IS the success
    rate - the single most useful number when comparing agents.
    """
    n_actions = environment.action_space.n
    total = 0.0
    for _ in range(episodes):
        state, info = environment.reset()
        done = False
        while not done:
            p = policy(state)
            action = np.random.choice(n_actions, p=p) if isinstance(p, np.ndarray) else int(p)
            state, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated
            total += reward
    return total / episodes


# =============================================================================
# Tabular plots
# =============================================================================
def plot_values(state_values, frame=None, env=None, cmap='coolwarm', title="State values V(s)"):
    """Heatmap of the state-value function V(s).

    state_values : flat (n_states,) or grid (side, side) array.
    frame        : optional env.render() image, drawn beside the heatmap.
    env          : optional; if given, holes/gift are labelled H / G instead of numbers.
    """
    grid = _to_grid_v(state_values)
    side = grid.shape[0]
    letters = _desc_letters(env, side)

    # Annotations: the value, or the terminal letter when we know the map.
    annot = np.empty((side, side), dtype=object)
    for r in range(side):
        for c in range(side):
            if letters is not None and letters[r, c] in 'HG':
                annot[r, c] = letters[r, c]
            else:
                annot[r, c] = f"{grid[r, c]:.2f}"

    ncols = 2 if frame is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    ax0 = axes[0] if frame is not None else axes
    sns.heatmap(grid, annot=annot, fmt='', cmap=cmap, cbar=False,
                annot_kws={'weight': 'bold', 'size': 12}, linewidths=2, ax=ax0)
    ax0.axis('off')
    if frame is not None:
        axes[1].imshow(frame)
        axes[1].axis('off')
    plt.suptitle(title, size=18)
    plt.tight_layout()
    plt.show()


def plot_policy(probs_or_qvals, frame=None, env=None, action_meanings=None,
                cmap='coolwarm', title="Policy"):
    """Show the greedy action letter in every tile.

    probs_or_qvals : flat (n_states, n_actions) or grid (side, side, n_actions);
                     works for a policy table AND for a Q table (argmax of either
                     is the greedy action).
    frame          : optional env.render() image, drawn beside the policy.
    env            : optional; if given, holes/gift show H / G instead of an action.
    """
    if action_meanings is None:
        action_meanings = FROZENLAKE_ACTIONS
    grid = _to_grid_q(probs_or_qvals)
    side = grid.shape[0]
    letters = _desc_letters(env, side)

    best = grid.argmax(axis=-1)                                  # greedy action per tile
    annot = np.empty((side, side), dtype=object)
    for r in range(side):
        for c in range(side):
            if letters is not None and letters[r, c] in 'HG':
                annot[r, c] = letters[r, c]
            else:
                annot[r, c] = action_meanings[int(best[r, c])]

    ncols = 2 if frame is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    ax0 = axes[0] if frame is not None else axes
    sns.heatmap(best, annot=annot, fmt='', cbar=False, cmap=cmap,
                annot_kws={'weight': 'bold', 'size': 12}, linewidths=2, ax=ax0)
    ax0.axis('off')
    if frame is not None:
        axes[1].imshow(frame)
        axes[1].axis('off')
    plt.suptitle(title, size=18)
    plt.tight_layout()
    plt.show()


def plot_action_values(action_values, env=None, cmap='coolwarm', title="Action values Q(s,a)"):
    """Draw all Q(s,a) values inside each tile as a four-triangle 'quatromatrix'.

    Each tile is split into 4 triangles - left, down, right, up - coloured and
    annotated with the corresponding Q value. action_values may be a flat
    (n_states, 4) table or a (side, side, 4) grid. Any grid size works: the
    triangle geometry and the text positions are generated, not hard-coded.
    """
    q = _to_grid_q(action_values)
    side = q.shape[0]
    letters = _desc_letters(env, side)

    fig, ax = plt.subplots(figsize=(1.6 * side, 1.6 * side))
    tripcolor = quatromatrix(q, ax=ax,
                             triplotkw={"color": "k", "lw": 1},
                             tripcolorkw={"cmap": cmap})
    ax.margins(0)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(tripcolor)

    # Text positions inside each cell, relative to its lower-left corner.
    # (Plot y grows upward, so the top row of the table is drawn at the top.)
    offsets = {0: (0.05, 0.45),    # LEFT  triangle
               1: (0.38, 0.12),    # DOWN  triangle
               2: (0.62, 0.45),    # RIGHT triangle
               3: (0.38, 0.78)}    # UP    triangle
    for r in range(side):
        for c in range(side):
            y0 = side - 1 - r                      # flip: table row 0 at the top of the plot
            if letters is not None and letters[r, c] in 'HG':
                ax.text(c + 0.45, y0 + 0.45, letters[r, c],
                        size=14, color="w", weight="bold")
                continue
            for a, (dx, dy) in offsets.items():
                ax.text(c + dx, y0 + dy, f"{q[r, c, a]:.2f}",
                        size=7, color="w", weight="bold")

    ax.set_title(title, size=18)
    plt.tight_layout()
    plt.show()


def quatromatrix(action_values, ax=None, triplotkw=None, tripcolorkw=None):
    """Helper for plot_action_values: split every cell into 4 coloured triangles.

    Generalized from the original 5x5-only version: the grid size is read from
    the input array, so it works for 4x4, 8x8, or any square map.
    """
    action_values = np.flipud(action_values)          # draw table row 0 at the top
    n, m = action_values.shape[:2]
    a = np.array([[0, 0], [0, 1], [.5, .5], [1, 0], [1, 1]])     # 4 corners + centre
    tr = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4], [1, 2, 4]])  # 4 triangles at the centre
    A = np.zeros((n * m * 5, 2))
    Tr = np.zeros((n * m * 4, 3))
    for i in range(n):
        for j in range(m):
            k = i * m + j
            A[k * 5:(k + 1) * 5, :] = np.c_[a[:, 0] + j, a[:, 1] + i]
            Tr[k * 4:(k + 1) * 4, :] = tr + k * 5
    # Triangle order produced by `tr` is: left, bottom, right, top.
    # FrozenLake actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP -> same order, no shuffle needed.
    C = np.c_[action_values[:, :, 0].flatten(),      # left  triangle <- LEFT  (0)
              action_values[:, :, 1].flatten(),      # down  triangle <- DOWN  (1)
              action_values[:, :, 2].flatten(),      # right triangle <- RIGHT (2)
              action_values[:, :, 3].flatten()].flatten()   # up   <- UP (3)

    ax.triplot(A[:, 0], A[:, 1], Tr, **triplotkw)
    return ax.tripcolor(A[:, 0], A[:, 1], Tr, facecolors=C, **tripcolorkw)


def plot_tabular_cost_to_go(action_values, xlabel=None, ylabel=None):
    """Heatmap of the 'cost-to-go' (-max_a Q(s,a)) for a tabular value table."""
    q = _to_grid_q(action_values)
    plt.figure(figsize=(8, 8))
    cost_to_go = -q.max(axis=-1)
    plt.imshow(cost_to_go, cmap='jet')
    plt.title("Estimated cost-to-go", size=24)
    plt.xlabel(xlabel, size=18)
    plt.ylabel(ylabel, size=18)
    plt.xticks([]); plt.yticks([])
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# =============================================================================
# Deep-RL plots and utilities
# =============================================================================
def plot_stats(stats):
    """Plot smoothed training curves (e.g. returns, loss) from a {name: [values]} dict."""
    rows = len(stats)
    fig, ax = plt.subplots(rows, 1, figsize=(12, 6))
    for i, key in enumerate(stats):
        vals = stats[key]
        vals = [np.mean(vals[j - 10:j + 10]) for j in range(10, len(vals) - 10)]
        axis = ax[i] if rows > 1 else ax
        axis.plot(range(len(vals)), vals)
        axis.set_title(key, size=18)
    plt.tight_layout()
    plt.show()


def seed_everything(env, seed: int = 42) -> None:
    """Fix all random seeds so a training run is reproducible.

    Gymnasium removed env.seed(); seeding now happens through reset(seed=...),
    which fixes the env's RNG for the whole life of the environment.
    """
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)


def plot_cost_to_go(env, q_network, xlabel=None, ylabel=None):
    """3-D surface of the estimated cost-to-go over a continuous 2-D state space."""
    if torch is None:
        raise ImportError("plot_cost_to_go requires PyTorch.")
    highx, highy = env.observation_space.high
    lowx, lowy = env.observation_space.low
    X = torch.linspace(lowx, highx, 100)
    Y = torch.linspace(lowy, highy, 100)
    X, Y = torch.meshgrid(X, Y, indexing='ij')
    q_net_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    Z = -q_network(q_net_input).max(dim=-1, keepdim=True)[0]
    Z = Z.reshape(100, 100).detach().numpy()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X.numpy(), Y.numpy(), Z, cmap='jet', linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(xlabel, size=14)
    ax.set_ylabel(ylabel, size=14)
    ax.set_title("Estimated cost-to-go", size=18)
    plt.tight_layout()
    plt.show()


def plot_max_q(env, q_network, xlabel=None, ylabel=None, action_labels=()):
    """Map of the greedy (argmax_a Q) action across a continuous 2-D state space."""
    if torch is None:
        raise ImportError("plot_max_q requires PyTorch.")
    import matplotlib.patches as mpatches
    highx, highy = env.observation_space.high
    lowx, lowy = env.observation_space.low
    X = torch.linspace(lowx, highx, 100)
    Y = torch.linspace(lowy, highy, 100)
    X, Y = torch.meshgrid(X, Y, indexing='ij')
    q_net_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    Z = q_network(q_net_input).argmax(dim=-1, keepdim=True)
    Z = Z.reshape(100, 100).T.detach().numpy()
    values = np.unique(Z.ravel()); values.sort()

    plt.figure(figsize=(5, 5))
    plt.xlabel(xlabel, size=14)
    plt.ylabel(ylabel, size=14)
    plt.title("Optimal action", size=18)
    im = plt.imshow(Z, cmap='jet')
    colors = [im.cmap(im.norm(v)) for v in values]
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, action_labels)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()


def plot_action_probs(probs, labels):
    """Bar chart of a policy's action probabilities pi(s) for one state."""
    plt.figure(figsize=(6, 4))
    plt.bar(labels, probs, color='orange')
    plt.title(r"$\pi(s)$", size=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
