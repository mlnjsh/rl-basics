# Reinforcement Learning — Course Notebooks

A hands-on, beginner-friendly tour of Reinforcement Learning, from Markov Decision
Processes all the way to deep policy-gradient methods. Every algorithm is taught on
the same tiny **5×5 Maze** environment so you can *see* what each method learns.

> The agent is drawn as a **fuchsia circle**; the goal is the green square.

## How the notebooks are organised

- **`envs.py`** — the `Maze` environment (the task every notebook solves).
- **`utils.py`** — all the plotting / evaluation helpers (`plot_policy`, `plot_values`,
  `plot_action_values`, `test_agent`, …).

Because of these two files, every lesson notebook starts with just:

```python
from envs import Maze
from utils import plot_policy, plot_values, plot_action_values, test_agent
```

instead of a giant setup cell. The only exceptions are the two **intro** notebooks
(`MDP_introduction` and `Classic_Control_Introduction`), which build everything inline
on purpose, so you can read the environment code itself.

Each lesson comes in two flavours:
- **starter** — code blanks left as exercises for you to fill in.
- **`_complete`** — the fully worked, fully commented solution.

## Run in Google Colab

Click a badge to open that notebook directly in Colab (no install needed — the first
cell pulls `envs.py` / `utils.py` and the pinned `gym` automatically).

| # | Lesson | Starter | Complete |
|---|--------|---------|----------|
| 0 | Markov Decision Process — intro | — | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/MDP_introduction.ipynb) |
| 0 | Classic Control (Gym) — intro | — | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Classic_Control_Introduction.ipynb) |
| 3 | Policy Iteration (DP) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_3_policy_iteration.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_3_policy_iteration_complete.ipynb) |
| 3 | Value Iteration (DP) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_3_value_iteration.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_3_value_iteration_complete.ipynb) |
| 4 | On-policy constant-α Monte Carlo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_4_on_policy_constant_alpha_mc.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_4_on_policy_constant_alpha_mc_complete.ipynb) |
| 4 | On-policy MC control | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_4_on_policy_control.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_4_on_policy_control_complete.ipynb) |
| 4 | Off-policy MC control | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_4_off_policy_control.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_4_off_policy_control_complete.ipynb) |
| 5 | SARSA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_5_sarsa.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_5_sarsa_complete.ipynb) |
| 5 | Q-Learning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_5_qlearning.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_5_qlearning_complete.ipynb) |
| 6 | n-step SARSA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_6_n_step_sarsa.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_6_n_step_sarsa_complete.ipynb) |
| 7 | Continuous observation spaces | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_7_continuous_observation_spaces.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_7_continuous_observation_spaces_complete.ipynb) |
| 8 | Deep SARSA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_8_deep_sarsa.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_8_deep_sarsa_complete.ipynb) |
| 9 | Deep Q-Learning (DQN) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_9_deep_q_learning.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_9_deep_q_learning_complete.ipynb) |
| 10 | REINFORCE (policy gradient) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_10_reinforce_CartPole.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_10_reinforce_CartPole_complete.ipynb) |
| 11 | Advantage Actor-Critic (A2C) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_11_advantage_actor_critic.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlnjsh/rl-basics/blob/main/Section_11_advantage_actor_critic_complete.ipynb) |

## Running locally

```bash
pip install gym==0.23.0 pygame numpy matplotlib seaborn torch tqdm
jupyter notebook
```

With `envs.py` and `utils.py` in the same folder as the notebooks, the `from envs import Maze`
and `from utils import …` lines work out of the box.
