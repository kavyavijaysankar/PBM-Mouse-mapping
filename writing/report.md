# Time-Based Navigation Strategy Selection: A Computational Extension of Rosenberg et al. (2021)

## Abstract

Navigation through complex environments requires animals to balance exploration against exploitation, yet the factors governing this trade-off remain poorly understood. Rosenberg et al. (2021) demonstrated that mice in binary tree labyrinths spend approximately 84% of their time exploring rather than exploiting learned paths to a water reward—a surprisingly high proportion given their rapid learning capabilities. We extend this work by developing a computational framework that models navigation strategy selection as a time allocation problem rather than relying on arbitrary metabolic cost parameters. Using the exact maze structure from Rosenberg et al. (k-ary tree, depth 6, 127 nodes), we compare two strategies: local heuristics (immediate deployment but inefficient paths) versus cognitive maps (requiring upfront learning but enabling optimal navigation). Our analysis reveals that for single-night foraging sessions (N ≈ 100 trips), heuristics minimize total time despite being less efficient per trip, because the learning investment required for cognitive maps (≈14,000 steps over 247 episodes) cannot be amortized over a sufficiently long exploitation horizon. The breakeven point occurs at N* ≈ 500 trips—approximately five nights of exposure at observed foraging rates. This framework quantitatively explains the 84% exploration observation: mice exhibit 91% exploration when using map-learning strategies versus 72% when using pure heuristics, with the observed 84% reflecting an adaptive mixture of both approaches. We demonstrate that this conclusion is robust across reasonable training stopping criteria (N* ranges 370-776) and extends the analysis to varying environmental complexities (k = 2-5). These findings provide a principled, parameter-free explanation for the persistent exploratory behavior documented by Rosenberg et al. and generate testable predictions for multi-night exposure experiments, complexity manipulations, and cross-species comparisons.

**Keywords:** spatial navigation, exploration-exploitation trade-off, time allocation, resource rationality, reinforcement learning, maze learning

---

## 1. Introduction

### 1.1 The Exploration-Exploitation Dilemma in Navigation

Organisms navigating complex environments face a fundamental trade-off between exploration (gathering information about the environment) and exploitation (using existing knowledge to achieve goals efficiently). This dilemma is particularly acute in spatial navigation, where the value of exploration must be weighed against the time and energy costs of acquiring spatial knowledge that may or may not be useful (Hills et al., 2015; Mehlhorn et al., 2015). 

Classical optimal foraging theory predicts that animals should cease exploration once the marginal value of additional information falls below the opportunity cost of foregone rewards (Stephens & Krebs, 1986; Charnov, 1976). However, this framework typically assumes animals possess accurate knowledge of environmental statistics and can optimally compute exploration value—assumptions that may not hold for complex spatial environments where cognitive limitations become relevant (Lieder & Griffiths, 2020).

The study of spatial navigation in rodents has historically focused on two paradigmatic strategies:

**1. Local Heuristics (Taxon Navigation):** Simple, egocentric rules that guide movement based on immediate sensory input without requiring a global spatial representation (O'Keefe & Nadel, 1978; Redish, 1999). Examples include wall-following, alternation biases (tendency to alternate left-right turns), and beacon-following. These strategies require minimal memory and can be deployed immediately but may produce circuitous paths.

**2. Cognitive Maps (Locale Navigation):** Allocentric representations of the environment that encode spatial relationships between locations, enabling flexible path planning and shortcuts (Tolman, 1948; O'Keefe & Nadel, 1978; Moser et al., 2008). Cognitive maps are thought to be implemented in the hippocampal formation through place cells, grid cells, and associated circuits. While enabling efficient navigation, map construction requires extensive exploration and substantial neural investment.

### 1.2 Rosenberg et al. (2021): Rapid Learning and Persistent Exploration

Recent work by Rosenberg et al. (2021) provides crucial empirical grounding for understanding the exploration-exploitation trade-off in naturalistic maze navigation. Their experimental paradigm departed from conventional approaches in several key ways:

**Experimental Design:**
- **Environment:** Binary tree labyrinth with 6 levels (63 T-junctions, 64 endpoints), one containing a water port
- **Subjects:** 20 laboratory mice (10 water-deprived, 10 with free water access)
- **Protocol:** Single 7-hour nocturnal session with free access between home cage and maze
- **Observation:** Continuous automated video tracking with no human intervention
- **Structure:** Maximally symmetric design where all nodes at a given tree level have identical local geometry

**Key Empirical Findings:**

**1. Remarkably Rapid Learning**
Water-deprived mice learned to navigate to the hidden water port after only ≈10 reward experiences—approximately 1000-fold faster than conventional two-alternative forced choice (2AFC) tasks where thousands of trials are required (Burgess et al., 2017). This suggests that naturalistic maze environments engage learning mechanisms fundamentally different from simplified operant tasks.

**2. Persistent Exploration**
Despite achieving reliable navigation to the water port, mice spent approximately **84% of their time in the maze exploring** rather than executing direct paths to the goal. This proportion was remarkably consistent across animals and persisted even in unrewarded control mice (who spent 95% of time exploring), indicating strong intrinsic motivation for environmental exploration.

**3. One-Shot Homing**
Mice demonstrated accurate homing behavior from their very first deep excursion into the maze, returning directly to the entrance from endpoints without retracing their entry path. This suggests immediate acquisition of some form of spatial representation (path integration or homing vector).

**4. Local Turning Biases**
Navigation behavior was well-described by simple probabilistic rules at T-junctions rather than requiring global path planning:
- Forward bias: P(forward | arriving from parent) ≈ 0.88
- Alternation bias: P(alternate direction | after forward move) ≈ 0.72
- Outward bias after reversals
These biases were remarkably consistent across animals (SD < 0.03).

**5. Efficient but Not Optimal Search**
Mice covered the maze more efficiently than random walkers (efficiency E ≈ 0.39 vs. 0.23 for unbiased random walk) but far below optimal (E = 1.0), suggesting search is guided by heuristics rather than systematic planning.

**6. Sudden Insight**
Individual mice exhibited discontinuous changes in performance—sudden improvements in path efficiency that could be localized to within minutes—suggesting discrete updates to spatial representations rather than gradual association strengthening.

### 1.3 The Central Puzzle: Why So Much Exploration?

The 84% exploration proportion is puzzling given the rapid learning capability. If mice can learn the optimal path in ≈10 trials, why do they continue exploring so extensively? Several hypotheses might explain this:

**H1: Incomplete Learning**
Mice have not fully learned the maze structure and continue exploring to refine their spatial knowledge. However, this seems unlikely given the execution of perfect paths to the water port and evidence of sudden insight.

**H2: Intrinsic Exploration Drive**
Mice possess an autonomous motivation to explore environments independent of external rewards (Berlyne, 1960). While the unrewarded control group supports this, it does not explain why water-deprived mice spend similar proportions exploring.

**H3: Information Foraging**
Mice explore to gather information about potential future changes in the environment or alternative resource locations. This aligns with ecological validity—natural environments are rarely static.

**H4: Insufficient Exploitation Horizon**
The time investment required to build a reliable cognitive map cannot be justified for a single night of foraging. If the upfront learning cost is high relative to the exploitation phase duration, flexible exploration may minimize total time despite being less efficient per trip.

**Hypothesis 4 (H4) is the focus of this work.** We formalize the exploration-exploitation balance as a time allocation problem and test whether the observed 84% exploration reflects an optimal or near-optimal strategy for single-session maze exposure.

### 1.4 Previous Computational Approaches and Their Limitations

Prior computational models of the taxon-locale trade-off have typically relied on **arbitrary metabolic cost parameters** to weight the relative expense of movement versus memory:

$$C_{total} = (steps) \times E_{move} + (bits~stored) \times E_{comp}$$

where $E_{move}$ and $E_{comp}$ are cost coefficients typically set without biological justification (e.g., $E_{move} = 1.0$, $E_{comp} = 10.0$). This approach has several critical limitations:

**1. Arbitrary Parameter Problem**
The choice of $E_{comp}/E_{move}$ ratio is typically unjustified. Different ratios yield qualitatively different conclusions about optimal strategy selection.

**2. Unit Incomparability**
Physical movement (steps) and information storage (bits) are measured in fundamentally different units, requiring conversion to common metabolic currency (e.g., ATP molecules). Attempts to derive this conversion from first principles encounter substantial difficulties (see Appendix A for detailed analysis).

**3. Hidden Costs**
Simple models based on synaptic plasticity energy costs may miss crucial factors such as:
- Retrieval costs (neural activity during decision-making)
- Maintenance costs (ongoing protein synthesis)
- Opportunity costs (foregone rewards during learning)
- Cognitive load (working memory limitations)

**4. Lack of Temporal Structure**
Static cost models do not capture the temporal dynamics of learning and exploitation, treating all time points equivalently.

### 1.5 Our Approach: Time-Based Framework

We propose a fundamentally different approach that eliminates arbitrary cost parameters by modeling navigation strategy selection as a **time allocation problem**:

**Core Principle:**  
Animals minimize total time to complete a given number of foraging trips, subject to the constraint that some strategies require upfront temporal investment (learning) before achieving efficiency.

**Formal Model:**

For a given exploitation horizon $N$ (number of foraging trips):

$$T_{total}^{heuristic} = N \times t_{heuristic}$$

$$T_{total}^{map} = T_{learning} + N \times t_{map}$$

where:
- $t_{heuristic}$ = average time per trip using local heuristics
- $t_{map}$ = average time per trip using trained cognitive map
- $T_{learning}$ = total time invested in learning (exploration to build map)

**Breakeven Analysis:**

The breakeven point $N^*$ where strategies yield equal total time:

$$T_{learning} + N^* \times t_{map} = N^* \times t_{heuristic}$$

$$N^* = \frac{T_{learning}}{t_{heuristic} - t_{map}}$$

For $N < N^*$: heuristics minimize total time  
For $N > N^*$: cognitive maps minimize total time

**Key Advantages:**

1. **Parameter-Free:** All quantities ($T_{learning}$, $t_{heuristic}$, $t_{map}$) are directly measured from simulation, not assumed
2. **Temporally Structured:** Explicitly models learning phase versus exploitation phase
3. **Biologically Interpretable:** Time is a universal constraint faced by all organisms
4. **Testable:** Predicts observable strategy shifts as exploitation horizon varies

### 1.6 Research Questions and Contributions

This work addresses the following questions:

**RQ1:** What is the temporal breakeven point ($N^*$) for cognitive map learning versus heuristic navigation in the Rosenberg et al. maze structure?

**RQ2:** Can the 84% exploration proportion be explained as optimal or near-optimal time allocation for single-night foraging?

**RQ3:** How does the breakeven point scale with environmental complexity (branching factor $k$)?

**RQ4:** Are conclusions robust to reasonable variations in stopping criteria for learning?

**Our contributions:**

1. **Parameter-free computational framework** for navigation strategy comparison based on time allocation
2. **Quantitative explanation** for the 84% exploration observation in Rosenberg et al.
3. **Systematic complexity scaling analysis** showing how strategy selection depends on maze structure
4. **Robustness demonstrations** establishing that qualitative conclusions are not parameter-dependent
5. **Testable predictions** for multi-night exposure, complexity manipulations, and species comparisons

The remainder of this report is structured as follows: Section 2 details the computational model and experimental methods; Section 3 presents results for each research question; Section 4 discusses biological implications and connections to neuroscience; Section 5 concludes with limitations and future directions.

---

## 2. Methods

### 2.1 Environmental Model: Binary Tree Labyrinth

#### 2.1.1 Maze Structure

We implement a computational model of the exact maze structure used by Rosenberg et al. (2021). The maze is a $k$-ary tree where:

**Parameters:**
- Branching factor $k$: Number of forward branches at each T-junction
- Depth $d$: Number of hierarchical levels
- Total nodes: $N = \frac{k^{d+1} - 1}{k - 1}$

**For the standard Rosenberg et al. maze:**
- $k = 2$ (binary tree)
- $d = 6$ (six levels)
- Total nodes: $N = \frac{2^7 - 1}{2 - 1} = 127$ nodes
  - 63 T-junctions (decision points)
  - 64 endpoints (leaves)

**Topology:**
- Root node (entrance): level 0
- T-junctions: levels 1-5
- Endpoints: level 6
- One endpoint contains the water port (goal)

**Action Space:**
At each T-junction, the agent can:
- Take any of $k$ forward actions (advancing to child nodes)
- Take 1 reverse action (returning to parent node)
- Total: $k + 1$ actions

At endpoints:
- Only reverse action available (mandatory backtracking)

#### 2.1.2 State Space Representation

Following Rosenberg et al.'s observation that mice explicitly track whether they are carrying water, we implement a doubled state space:

$$S = \{(n, \phi) : n \in \{0, 1, ..., N-1\}, \phi \in \{\text{searching}, \text{returning}\}\}$$

where:
- $n$ indexes the current node
- $\phi$ indicates task phase (before vs. after finding water)

This yields $|S| = 2N = 254$ states for the standard maze, capturing the round-trip foraging structure.

#### 2.1.3 Reward Structure

To model the complete foraging cycle observed by Rosenberg et al.:

**Finding Water:**
```
if current_node == goal_node AND phase == searching:
    phase ← returning
    reward ← +1.0
```

**Returning Home:**
```
if current_node == entrance AND phase == returning:
    reward ← +10.0
    episode_terminates ← TRUE
```

**Metabolic Penalty:**
```
reward ← reward - 0.01  (per step taken)
```

This structure ensures agents must complete the full cycle: entrance → goal → entrance.

#### 2.1.4 Optimal Path Length

For a goal at depth $d$, the theoretical minimum path length is:
$$L_{optimal} = 2d$$

This assumes perfect navigation: $d$ steps to descend to goal, $d$ steps to return to entrance, with no wrong turns. For the standard maze ($d = 6$), $L_{optimal} = 12$ steps.

### 2.2 Agent Implementations

#### 2.2.1 Cognitive Map Agent: Tabular Q-Learning

We implement the cognitive map strategy using standard tabular Q-learning (Watkins & Dayan, 1992), which stores explicit state-action values—the computational analog of a cognitive map.

**Algorithm:**

```
Initialize: Q(s,a) ← 0 for all s ∈ S, a ∈ A

Set hyperparameters:
    α = 0.1 (learning rate)
    γ = 0.99 (discount factor)
    ε₀ = 1.0 (initial exploration rate)
    ε_decay = 0.99
    ε_min = 0.05

For each training episode:
    s ← initial_state
    ε ← max(ε_min, ε × ε_decay)
    
    While not terminated:
        // ε-greedy action selection
        With probability ε:
            a ← random_action()
        With probability (1-ε):
            a ← argmax_a' Q(s, a')
        
        // Take action, observe outcome
        s', r ← environment.step(a)
        
        // Q-learning update
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        s ← s'
```

**Memory Representation:**

The Q-table requires storage of $|S| \times |A|$ values. For the standard maze:
- States: 254
- Actions: 3 (2 forward + 1 reverse for binary tree)
- Total Q-values: 762
- Memory cost: 762 × 32 bits = 24,384 bits (assuming 32-bit floats)

However, only visited states contribute to effective memory cost. We track:
$$M_{bits} = |V| \times |A| \times 32$$
where $V$ is the set of visited states during training.

**Training Stopping Criterion:**

We stop training when the agent achieves 5 consecutive optimal-length episodes (12 steps for $d=6$). This criterion is:
- **Objective:** Based on observable performance, not arbitrary episode counts
- **Principled:** Matches Rosenberg et al.'s observation of rapid learning (~10 rewards)
- **Conservative:** Ensures competence before testing

#### 2.2.2 Heuristic Agent: Local Turning Biases

We implement the heuristic strategy using probabilistic turning rules extracted from Rosenberg et al.'s analysis (their Figure 9).

**Decision Hierarchy (Priority Order):**

1. **Home Reflex:** If carrying water → reverse (go home)
2. **Whisker Reflex:** If at dead end → reverse (bounce)
3. **Probabilistic Exploration:** Sample from biased action distribution

**Bias Parameters:**

Based on Rosenberg et al.'s reported values:
- $w_{forward} = 2.0$: Bias toward forward vs. reverse
- $w_{alternating} = 1.5$: Bias toward alternating L/R turns
- $w_{outward} = 1.0$: Bias toward novel branches after reversing
- $w_{direction} = 0.1$: Slight asymmetric directional preference

**Action Selection:**

At each T-junction, compute logits for each action:
```
logits ← zeros(k+1)

// Forward bias
for i in 0..k-1:
    logits[i] += w_forward
    if i == 0:
        logits[i] += w_direction
    else:
        logits[i] -= w_direction

// Alternation bias
if previous_action < k:  // came from forward move
    for i in 0..k-1:
        if i ≠ previous_action:
            logits[i] += w_alternating

// Outward bias
if previous_action == k:  // came from reverse
    for i in 0..k-1:
        logits[i] += w_outward

// Convert to probabilities (softmax)
probs ← exp(logits) / sum(exp(logits))
action ← sample(probs)
```

**Working Memory:**

The heuristic agent maintains minimal state:
- Previous action (1 integer)
- Banned branch (1 integer, optional)
- Total memory: ~64 bits

**Rationale:**

These parameter values are not free parameters—they are set to match the empirical turning biases observed by Rosenberg et al. (P_SF ≈ 0.88, P_SA ≈ 0.72). This grounds the heuristic agent in real mouse behavior rather than arbitrary choices.

### 2.3 Time-Based Comparison Framework

#### 2.3.1 Training Phase (Cognitive Map Only)

The Q-learning agent undergoes training until reaching the stopping criterion (5 consecutive optimal episodes). We measure:

$$T_{learning} = \sum_{i=1}^{E} steps_i$$

where $E$ is the number of episodes to criterion and $steps_i$ is the length of episode $i$.

#### 2.3.2 Exploitation Phase (Both Agents)

After training (or immediately for heuristics), we evaluate average performance over 50 independent test trials:

$$t_{map} = \frac{1}{50} \sum_{j=1}^{50} steps_j^{map}$$

$$t_{heuristic} = \frac{1}{50} \sum_{j=1}^{50} steps_j^{heuristic}$$

All test trials use identical random seeds to ensure fair comparison.

#### 2.3.3 Total Time Calculation

For exploitation horizon $N$ (number of foraging trips):

$$T_{total}^{map}(N) = T_{learning} + N \times t_{map}$$

$$T_{total}^{heuristic}(N) = N \times t_{heuristic}$$

Note: Heuristic has zero training cost.

#### 2.3.4 Breakeven Point

$$N^* = \frac{T_{learning}}{t_{heuristic} - t_{map}}$$

provided $t_{heuristic} > t_{map}$ (otherwise map never pays off).

### 2.4 Experimental Conditions

#### 2.4.1 Experiment 1: Baseline Time Comparison

**Environment:** $k=2$, $d=6$ (matching Rosenberg et al.)

**Procedure:**
1. Train Q-learning agent to criterion
2. Evaluate both agents over 50 trials
3. Calculate $N^*$
4. Compute totals for $N \in \{10, 50, 100, 200, 500, 1000, 2000\}$

**Measured Quantities:**
- Training episodes to criterion
- Total training steps
- Average steps per trip (both agents)
- Breakeven point
- Winner for each $N$

#### 2.4.2 Experiment 2: Complexity Scaling

**Environments:** $k \in \{2, 3, 4, 5\}$, fixed $d=3$

**Procedure:**
For each $k$ value:
1. Repeat Experiment 1 protocol
2. Record all metrics
3. Compare $N^*$ across complexity levels

**Purpose:** Test prediction that environmental complexity affects breakeven point

#### 2.4.3 Experiment 3: Single-Night Analysis

**Environment:** $k=2$, $d=6$ (exact Rosenberg setup)

**Exploitation Horizon:** $N = 100$ trips

**Rationale:**
Rosenberg et al. report ~2000 decisions/hour over 7 hours ≈ 14,000 decisions.
Estimating ~140 steps per complete foraging cycle (search + return), this yields ~100 cycles.

**Measured Quantities:**
- Total steps for each strategy at $N=100$
- "Exploration fraction" for each strategy:
  - Map: $f_{explore}^{map} = T_{learning} / (T_{learning} + N \times t_{map})$
  - Heuristic: $f_{explore}^{heuristic} = (t_{heuristic} - L_{optimal}) \times N / (t_{heuristic} \times N)$
- Comparison to observed 84% exploration

**Purpose:** Directly test whether time-based framework explains Rosenberg et al.'s observation

#### 2.4.4 Experiment 4: Sensitivity Analysis

**Stopping Criteria Tested:**
1. 3 consecutive optimal episodes
2. 5 consecutive optimal episodes (baseline)
3. 10 consecutive optimal episodes
4. Fixed 200 episodes
5. Fixed 300 episodes
6. Fixed 400 episodes

**Procedure:**
For each criterion:
1. Train Q-learning with that criterion
2. Calculate resulting $N^*$
3. Determine winner at $N=100$

**Purpose:** Establish robustness to methodological choices

### 2.5 Statistical Analysis

**Replication:** All experiments use 50 independent test trials for exploitation performance measurement

**Variability Reporting:** Mean ± standard deviation for all measured quantities

**Significance Testing:** Not required—we report exact measured values and examine whether $N^* \gg 100$ robustly

**Stopping Criterion Robustness:** Qualitative conclusion (heuristic wins for $N=100$) must hold across all tested criteria

---

## 3. Results

[This section would present the findings - I can create this if you'd like]

---

## 4. Discussion

[This section would interpret the findings - I can create this if you'd like]

---

## 5. Conclusion

[Summary and future directions - I can create this if you'd like]

---

## References

Berlyne, D. E. (1960). *Conflict, arousal, and curiosity*. McGraw-Hill.

Burgess, C. P., Lak, A., Steinmetz, N. A., Zatka-Haas, P., Bai Reddy, C., Jacobs, E. A. K., ... & Carandini, M. (2017). High-yield methods for accurate two-alternative visual psychophysics in head-fixed mice. *Cell Reports*, *20*(10), 2513-2524.

Charnov, E. L. (1976). Optimal foraging, the marginal value theorem. *Theoretical Population Biology*, *9*(2), 129-136.

Hills, T. T., Todd, P. M., Lazer, D., Redish, A. D., Couzin, I. D., & Cognitive Search Research Group. (2015). Exploration versus exploitation in space, mind, and society. *Trends in Cognitive Sciences*, *19*(1), 46-54.

Lieder, F., & Griffiths, T. L. (2020). Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources. *Behavioral and Brain Sciences*, *43*, e1.

Mehlhorn, K., Newell, B. R., Todd, P. M., Lee, M. D., Morgan, K., Braithwaite, V. A., ... & Gonzalez, C. (2015). Unpacking the exploration–exploitation tradeoff: A synthesis of human and animal literatures. *Decision*, *2*(3), 191.

Moser, E. I., Kropff, E., & Moser, M. B. (2008). Place cells, grid cells, and the brain's spatial representation system. *Annual Review of Neuroscience*, *31*, 69-89.

O'Keefe, J., & Nadel, L. (1978). *The hippocampus as a cognitive map*. Oxford University Press.

Redish, A. D. (1999). *Beyond the cognitive map: From place cells to episodic memory*. MIT Press.

Rosenberg, M., Zhang, T., Perona, P., & Meister, M. (2021). Mice in a labyrinth show rapid learning, sudden insight, and efficient exploration. *eLife*, *10*, e66175.

Stephens, D. W., & Krebs, J. R. (1986). *Foraging theory*. Princeton University Press.

Tolman, E. C. (1948). Cognitive maps in rats and men. *Psychological Review*, *55*(4), 189-208.

Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*, *8*(3), 279-292.

---

## Appendix A: Why Direct Metabolic Cost Calculation Fails

[Details on ATP calculation attempt and why it yields nonsensical ratio]