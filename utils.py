"""utils.py - Plotting and evaluation helpers shared by all RL notebooks.

Every notebook (except the two intro notebooks) imports only what it needs, e.g.:

    from utils import plot_policy, plot_values, plot_action_values, test_agent

This file is the *superset* of every helper used across the course:

  Tabular notebooks (Sections 3-7)
      plot_policy, plot_values, plot_action_values, quatromatrix,
      plot_tabular_cost_to_go, test_agent
  Deep-RL notebooks (Sections 8-9)
      plot_stats, plot_cost_to_go, plot_max_q, seed_everything, test_agent
  Policy-gradient notebooks (Sections 10-11)
      plot_action_probs, plot_stats, test_policy_network, seed_everything
  Intro
      display_video, test_env

`torch` and `matplotlib.patches` are imported lazily (only inside the deep-RL
functions) so the early tabular notebooks can `import utils` even on a machine
without PyTorch installed.
"""

import numpy as np                       # Arrays for value/Q tables and image frames.
import matplotlib                        # We toggle the backend when rendering videos.
import matplotlib.pyplot as plt          # All static plots (heatmaps, surfaces, bars).
import seaborn as sns                    # Pretty annotated heatmaps for policies and state values.
from matplotlib import animation         # Turns a list of frames into an animation.
from IPython.display import HTML         # Embeds that animation in the notebook output.

import gym                               # Only used for the type hint in seed_everything.

# --- Lazy heavy imports -------------------------------------------------------
# torch and matplotlib.patches are needed only by the deep-RL helpers below.
# Importing them lazily keeps `import utils` working in torch-free environments.
try:
    import torch
except ImportError:                      # pragma: no cover - torch is optional for tabular notebooks.
    torch = None


# =============================================================================
# Video / rollout helpers
# =============================================================================
def display_video(frames):
    """Turn a list of RGB frames into an HTML5 video the notebook can play inline."""
    # Copied from: https://colab.research.google.com/github/deepmind/dm_control/blob/master/tutorial.ipynb
    orig_backend = matplotlib.get_backend()      # Remember the interactive backend.
    matplotlib.use('Agg')                        # Switch to a non-GUI backend to build the frames.
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    matplotlib.use(orig_backend)                 # Restore the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])                     # Show the first frame.
    def update(frame):
        im.set_data(frame)                        # Each animation tick swaps in the next frame.
        return [im]
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                    interval=50, blit=True, repeat=False)
    return HTML(anim.to_html5_video())


def test_agent(environment, policy, episodes=10):
    """Run a trained tabular `policy` for several episodes and return a playable video.

    `policy(state)` may return either a single action (greedy) or a probability
    vector over the 4 actions (stochastic) - both cases are handled below.
    """
    frames = []
    for episode in range(episodes):
        state = environment.reset()
        done = False
        frames.append(environment.render(mode="rgb_array"))   # Capture the starting frame.

        while not done:
            p = policy(state)
            if isinstance(p, np.ndarray):
                action = np.random.choice(4, p=p)             # Stochastic policy -> sample an action.
            else:
                action = p                                    # Deterministic policy -> use it directly.
            next_state, reward, done, extra_info = environment.step(action)
            frames.append(environment.render(mode="rgb_array"))
            state = next_state

    return display_video(frames)


def test_env(environment, episodes=10):
    """Run a RANDOM agent (action_space.sample()) - used in the intro to see an untrained env."""
    frames = []
    for episode in range(episodes):
        state = environment.reset()
        done = False
        frames.append(environment.render(mode="rgb_array"))

        while not done:
            action = environment.action_space.sample()        # Pick a uniformly random legal action.
            next_state, reward, done, extra_info = environment.step(action)
            frames.append(environment.render(mode="rgb_array"))
            state = next_state

    return display_video(frames)


def test_policy_network(env, policy, episodes=10):
    """Roll out a neural-network policy (Sections 10-11) and return a playable video.

    Here `policy` is a torch network: it maps a state tensor to action probabilities,
    from which we sample with `.multinomial`.
    """
    frames = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        frames.append(env.render(mode="rgb_array"))

        while not done:
            state = torch.from_numpy(state).unsqueeze(0).float()  # numpy state -> batched float tensor.
            action = policy(state).multinomial(1).item()          # Sample one action from the policy.
            next_state, _, done, _ = env.step(action)
            frames.append(env.render(mode="rgb_array"))
            state = next_state

    return display_video(frames)


# =============================================================================
# Tabular plots (Sections 3-7)
# =============================================================================
def plot_policy(probs_or_qvals, frame, action_meanings=None):
    """Draw the greedy action in every grid square next to a picture of the maze."""
    if action_meanings is None:
        action_meanings = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    max_prob_actions = probs_or_qvals.argmax(axis=-1)             # Best action index per square.
    probs_copy = max_prob_actions.copy().astype(object)
    for key in action_meanings:
        probs_copy[probs_copy == key] = action_meanings[key]     # Replace indices with letters U/R/D/L.
    sns.heatmap(max_prob_actions, annot=probs_copy, fmt='', cbar=False, cmap='coolwarm',
                annot_kws={'weight': 'bold', 'size': 12}, linewidths=2, ax=axes[0])
    axes[1].imshow(frame)
    axes[0].axis('off')
    axes[1].axis('off')
    plt.suptitle("Policy", size=18)
    plt.tight_layout()


def plot_values(state_values, frame):
    """Draw a heatmap of the state-value function V(s) next to the maze."""
    f, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(state_values, annot=True, fmt=".2f", cmap='coolwarm',
                annot_kws={'weight': 'bold', 'size': 12}, linewidths=2, ax=axes[0])
    axes[1].imshow(frame)
    axes[0].axis('off')
    axes[1].axis('off')
    plt.tight_layout()


def plot_action_values(action_values):
    """Draw all four Q(s,a) values inside each grid square as a four-triangle 'quatromatrix'."""
    # Pixel positions for the four action labels (up/right/down/left text) inside each cell.
    text_positions = [
        [(0.35, 4.75), (1.35, 4.75), (2.35, 4.75), (3.35, 4.75), (4.35, 4.75),
         (0.35, 3.75), (1.35, 3.75), (2.35, 3.75), (3.35, 3.75), (4.35, 3.75),
         (0.35, 2.75), (1.35, 2.75), (2.35, 2.75), (3.35, 2.75), (4.35, 2.75),
         (0.35, 1.75), (1.35, 1.75), (2.35, 1.75), (3.35, 1.75), (4.35, 1.75),
         (0.35, 0.75), (1.35, 0.75), (2.35, 0.75), (3.35, 0.75), (4.35, 0.75)],
        [(0.6, 4.45), (1.6, 4.45), (2.6, 4.45), (3.6, 4.45), (4.6, 4.45),
         (0.6, 3.45), (1.6, 3.45), (2.6, 3.45), (3.6, 3.45), (4.6, 3.45),
         (0.6, 2.45), (1.6, 2.45), (2.6, 2.45), (3.6, 2.45), (4.6, 2.45),
         (0.6, 1.45), (1.6, 1.45), (2.6, 1.45), (3.6, 1.45), (4.6, 1.45),
         (0.6, 0.45), (1.6, 0.45), (2.6, 0.45), (3.6, 0.45), (4.6, 0.45)],
        [(0.35, 4.15), (1.35, 4.15), (2.35, 4.15), (3.35, 4.15), (4.35, 4.15),
         (0.35, 3.15), (1.35, 3.15), (2.35, 3.15), (3.35, 3.15), (4.35, 3.15),
         (0.35, 2.15), (1.35, 2.15), (2.35, 2.15), (3.35, 2.15), (4.35, 2.15),
         (0.35, 1.15), (1.35, 1.15), (2.35, 1.15), (3.35, 1.15), (4.35, 1.15),
         (0.35, 0.15), (1.35, 0.15), (2.35, 0.15), (3.35, 0.15), (4.35, 0.15)],
        [(0.05, 4.45), (1.05, 4.45), (2.05, 4.45), (3.05, 4.45), (4.05, 4.45),
         (0.05, 3.45), (1.05, 3.45), (2.05, 3.45), (3.05, 3.45), (4.05, 3.45),
         (0.05, 2.45), (1.05, 2.45), (2.05, 2.45), (3.05, 2.45), (4.05, 2.45),
         (0.05, 1.45), (1.05, 1.45), (2.05, 1.45), (3.05, 1.45), (4.05, 1.45),
         (0.05, 0.45), (1.05, 0.45), (2.05, 0.45), (3.05, 0.45), (4.05, 0.45)]]

    fig, ax = plt.subplots(figsize=(7, 7))
    tripcolor = quatromatrix(action_values, ax=ax,
                             triplotkw={"color": "k", "lw": 1}, tripcolorkw={"cmap": "coolwarm"})
    ax.margins(0)
    ax.set_aspect("equal")
    fig.colorbar(tripcolor)

    # Write the numeric Q-value into each of the four triangles of every cell.
    for j, av in enumerate(text_positions):
        for i, (xi, yi) in enumerate(av):
            plt.text(xi, yi, round(action_values[:, :, j].flatten()[i], 2), size=8, color="w", weight="bold")

    plt.title("Action values Q(s,a)", size=18)
    plt.tight_layout()
    plt.show()


def quatromatrix(action_values, ax=None, triplotkw=None, tripcolorkw=None):
    """Helper for plot_action_values: split each grid cell into 4 coloured triangles (one per action)."""
    action_values = np.flipud(action_values)         # Flip so row 0 is drawn at the top.
    n = 5
    m = 5
    a = np.array([[0, 0], [0, 1], [.5, .5], [1, 0], [1, 1]])   # 5 vertices: 4 corners + centre.
    tr = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4], [1, 2, 4]])  # 4 triangles meeting at the centre.
    A = np.zeros((n * m * 5, 2))
    Tr = np.zeros((n * m * 4, 3))
    for i in range(n):
        for j in range(m):
            k = i * m + j
            A[k * 5:(k + 1) * 5, :] = np.c_[a[:, 0] + j, a[:, 1] + i]   # Shift the 5 vertices into cell (i, j).
            Tr[k * 4:(k + 1) * 4, :] = tr + k * 5                       # Offset triangle indices for this cell.
    # Colour order per triangle = left, down, right, up (matches the on-screen layout).
    C = np.c_[action_values[:, :, 3].flatten(), action_values[:, :, 2].flatten(),
              action_values[:, :, 1].flatten(), action_values[:, :, 0].flatten()].flatten()

    ax.triplot(A[:, 0], A[:, 1], Tr, **triplotkw)
    tripcolor = ax.tripcolor(A[:, 0], A[:, 1], Tr, facecolors=C, **tripcolorkw)
    return tripcolor


def plot_tabular_cost_to_go(action_values, xlabel, ylabel):
    """Heatmap of the 'cost-to-go' (-max_a Q(s,a)) for a tabular/tile-coded value table (Section 7)."""
    plt.figure(figsize=(8, 8))
    cost_to_go = -action_values.max(axis=-1)         # Cost-to-go = negative of the best action value.
    plt.imshow(cost_to_go, cmap='jet')
    plt.title("Estimated cost-to-go", size=24)
    plt.xlabel(xlabel, size=18)
    plt.ylabel(ylabel, size=18)
    plt.xticks([])
    plt.yticks([])
    plt.xticks()
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# =============================================================================
# Deep-RL plots and utilities (Sections 8-11)
# =============================================================================
def plot_stats(stats):
    """Plot smoothed training curves (e.g. returns, loss) stored in a {name: [values]} dict."""
    rows = len(stats)
    cols = 1

    fig, ax = plt.subplots(rows, cols, figsize=(12, 6))

    for i, key in enumerate(stats):
        vals = stats[key]
        # Smooth with a centred moving average so the noisy per-episode curve is readable.
        vals = [np.mean(vals[i-10:i+10]) for i in range(10, len(vals)-10)]
        if len(stats) > 1:
            ax[i].plot(range(len(vals)), vals)
            ax[i].set_title(key, size=18)
        else:
            ax.plot(range(len(vals)), vals)
            ax.set_title(key, size=18)
    plt.tight_layout()
    plt.show()


def seed_everything(env: gym.Env, seed: int = 42) -> None:
    """Fix all random seeds (env, numpy, torch) so a training run is reproducible."""
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)   # Force deterministic CUDA/CPU kernels where possible.


def plot_cost_to_go(env, q_network, xlabel=None, ylabel=None):
    """3-D surface of the estimated cost-to-go over a continuous 2-D state space (Sections 8-9)."""
    highx, highy = env.observation_space.high        # Upper bounds of the 2-D observation.
    lowx, lowy = env.observation_space.low           # Lower bounds.
    X = torch.linspace(lowx, highx, 100)
    Y = torch.linspace(lowy, highy, 100)
    X, Y = torch.meshgrid(X, Y)                       # Build a 100x100 grid of states.

    q_net_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    Z = - q_network(q_net_input).max(dim=-1, keepdim=True)[0]   # cost-to-go = -max_a Q(s,a).
    Z = Z.reshape(100, 100).detach().numpy()
    X = X.numpy()
    Y = Y.numpy()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='jet', linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(xlabel, size=14)
    ax.set_ylabel(ylabel, size=14)
    ax.set_title("Estimated cost-to-go", size=18)
    plt.tight_layout()
    plt.show()


def plot_max_q(env, q_network, xlabel=None, ylabel=None, action_labels=[]):
    """Map of the greedy (argmax_a Q) action across a continuous 2-D state space."""
    import matplotlib.patches as mpatches            # Lazy import: only needed for the legend here.
    highx, highy = env.observation_space.high
    lowx, lowy = env.observation_space.low
    X = torch.linspace(lowx, highx, 100)
    Y = torch.linspace(lowy, highy, 100)
    X, Y = torch.meshgrid(X, Y)
    q_net_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    Z = q_network(q_net_input).argmax(dim=-1, keepdim=True)     # Best action index at each grid state.
    Z = Z.reshape(100, 100).T.detach().numpy()
    values = np.unique(Z.ravel())
    values.sort()

    plt.figure(figsize=(5, 5))
    plt.xlabel(xlabel, size=14)
    plt.ylabel(ylabel, size=14)
    plt.title("Optimal action", size=18)

    im = plt.imshow(Z, cmap='jet')
    colors = [im.cmap(im.norm(value)) for value in values]      # One colour per distinct action.
    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, action_labels)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()


def plot_action_probs(probs, labels):
    """Bar chart of a policy's action probabilities pi(s) for one state (Sections 10-11)."""
    plt.figure(figsize=(6, 4))
    plt.bar(labels, probs, color='orange')
    plt.title(r"$\pi(s)$", size=16)                  # Raw string so the LaTeX \pi renders cleanly.
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
