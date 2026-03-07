# Intermediate Memory Capacity Optimizes Spatial Navigation

## Overview

This project explores a question in spatial navigation: **how much memory capacity is required for efficient navigation and exploration?** 

Rosenberg et al. (2021) demonstrated that mice navigating a binary tree maze learned rapidly (after ~10 reward experiences) yet continued exploring 84% of the time rather than executing optimal paths. This persistent exploration contradicts expectations for animals with complete cognitive maps, who should minimize exploration once spatial knowledge is acquired.

**Hypothesis:** The 84% exploration reflects intermediate memory capacity, sufficient for systematic search and reliable homing, but insufficient for planning optimal routes.

**Key finding:** Minimal memory (20 nodes, 864 bits) outperforms unlimited memory by 20% in navigation efficiency, demonstrating that **forgetting is computationally beneficial, not merely a biological constraint**.

For complete analysis, results, and discussion, see [**report.pdf**](report.pdf).

---

## Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Dependencies
```bash
pip install numpy pandas matplotlib seaborn gymnasium
```

### Setup
```bash
git clone [repository-url]
cd memory-capacity-navigation
```

---

## Project Structure
```
.
├── memory_capacity_analysis.ipynb    # Main analysis notebook
├── memory_hierarchy_agents.py        # Agent implementations
├── labyrinth.py                      # Binary tree environment
├── figures/                          # Generated plots & report figures
│   ├── Branch Bias.png
│   ├── Navigation Efficiency and Exploration Coverage.png
│   └── Completion Rate vs Memory Level.png
├── report.pdf                        # Full report
└── README.md
```

---

## Quick Start

### Run the Full Analysis

Open and run `memory_capacity_analysis.ipynb`:
```bash
jupyter notebook memory_capacity_analysis.ipynb
```

The notebook contains:
1. **Environment setup** - Binary tree maze (k=2, depth=6)
2. **Agent definitions** - Six memory levels (L0-L2)
3. **Experiments** - 50 trials per condition
4. **Analysis & visualization** - Performance metrics and figures

All results and findings are documented in [**report.pdf**](report.pdf).

---

## Memory Hierarchy

We tested six agents with varying memory capacities:

| Level | Type | Memory (bits) | Description |
|-------|------|---------------|-------------|
| **L0** | Reactive | 224 | No spatial memory - pure heuristics |
| **L1a** | 20 nodes | 864 | Minimal memory buffer (FIFO) |
| **L1b** | 50 nodes | 1,824 | Small memory buffer |
| **L1c** | 100 nodes | 3,424 | Medium memory buffer |
| **L1d** | Unlimited | 4,288 | Unlimited memory within environment |
| **L2** | Full Memory | 24,384 | Complete Q-learning cognitive map |

### Navigation Strategy
- **Levels 0-1d:** Local heuristics (forward bias, alternation, branch preference) + novelty penalty
- **Level 2:** Tabular Q-learning trained to optimality

---

## Modifying Parameters

### Change Memory Capacities

Edit `memory_variants` in Cell 9:
```python
memory_variants = [
    {'name': 'Level 1a', 'max_nodes': 20},
    {'name': 'Level 1b', 'max_nodes': 50},
    {'name': 'Level 1c', 'max_nodes': 100},
    {'name': 'Level 1d', 'max_nodes': None},  # Unlimited
]
```

### Change Number of Trials

Modify `num_trials` in Cells 7, 8, 10:
```python
summary, _ = evaluate_agent(agent, env, 'reactive', num_trials=50)
```

### Change Novelty Penalty

In `memory_hierarchy_agents.py`, modify the penalty weight in the `MinimalMemoryAgent` class:
```python
penalty = 0.5 * np.log(self.visit_counts[next_node] + 1)  # Change 0.5 to test other values
```

### Change Maze Configuration

In the notebook, modify the environment parameters in the cell where `LabyrinthEnv` is created:
```python
# Change k (branching factor) and depth
env = LabyrinthEnv(k=2, depth=6)  # Default: binary tree, 6 levels deep

# Examples:
# env = LabyrinthEnv(k=3, depth=5)  # Ternary tree, 5 levels
# env = LabyrinthEnv(k=2, depth=7)  # Binary tree, 7 levels (larger maze)
```

**Note:** Changing k and depth will affect:
- Total number of nodes: approximately k^depth
- Task difficulty and completion rates
- Memory requirements for optimal navigation


---

## Environment Details

**Binary Tree Labyrinth:**
- k = 2 (binary branching)
- depth = 6 (6 levels deep)
- 63 T-junctions, 64 endpoints
- 1 water reward location
- 254 total states (doubled for search/return phases)

**Task:** Navigate from entrance → water port → entrance

**Performance metrics:**
- Task completion rate
- Steps per episode
- Tree coverage (% nodes visited)
- Exploration fraction

---

## Generating Figures

Run the notebook cells in order to reproduce all figures:

**Figure 1: Memory-Performance Relationship**
```python
# Cell 13: 2-panel plot
# Saved as: figures/Navigation Efficiency and Exploration Coverage.png
```

**Figure 2: Completion Rates**
```python
# Cell 15: Bar chart
# Saved as: figures/Completion Rate vs Memory Level.png
```

---

## Reference
```bibtex
@article{rosenberg2021mice,
  title={Mice in a labyrinth show rapid learning, sudden insight, and efficient exploration},
  author={Rosenberg, Mathew and Zhang, Tony and Perona, Pietro and Meister, Markus},
  journal={eLife},
  volume={10},
  pages={e66175},
  year={2021},
  publisher={eLife Sciences Publications Limited}
}
```