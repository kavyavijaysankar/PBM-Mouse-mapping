"""
Agent Implementations for Time-Based Navigation Experiments

Contains only the agents used in the time-based framework:
1. QLearning_agent - Tabular Q-learning (cognitive map strategy)
2. HeuristicAgent - Local turning biases (heuristic strategy)

No metabolic cost calculations - those are irrelevant for time-based comparison.
"""

import numpy as np


# =============================================================================
# Q-LEARNING AGENT (Cognitive Map)
# =============================================================================

class QLearning_agent:
    """
    Standard tabular Q-learning agent representing cognitive map strategy.
    
    Stores explicit state-action values (Q-table) which serve as a 
    computational cognitive map. After training, enables optimal navigation.
    """
    
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.05
        
        self.q_table = np.zeros((num_states, num_actions))
        
        # Track visited states for memory calculation
        self.visited_states = set()
        
        # Memory cost in bits (32-bit floats)
        self.memory_bits = num_states * num_actions * 32

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """Q-learning update rule."""
        self.visited_states.add(state)
        
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def get_dynamic_memory_bits(self):
        """Calculate memory based only on visited states."""
        return len(self.visited_states) * self.num_actions * 32


# =============================================================================
# HEURISTIC AGENT (Local Turning Biases)
# =============================================================================

class HeuristicAgent:
    """
    Local heuristic agent using turning biases from Rosenberg et al. (2021).
    
    Implements distinct navigation strategies for exploration vs. homing:
    
    EXPLORATION (searching for water):
        - Forward bias: P_SF ≈ 0.88 (prefer forward over reverse)
        - Alternation bias: P_SA ≈ 0.72 (prefer alternating L/R turns)
        - Branch bias (subtle): P_BS ≈ 0.54 (slight preference for turning)
        - Outward bias: prefer forward after reversing
    
    HOMING (returning with water):
        - Strong branch bias: Always prefer branches over straight
        - This implements the observed homing behavior where mice reliably
          return to entrance by preferring turns at T-junctions
    
    Parameters set to match Rosenberg et al. observations:
        w_forward = 2.2
        w_alternating = 1.8
        w_outward = 1.0
        w_branch_explore = 0.16 (subtle, ~54% turn probability)
        w_branch_home = 10.0 (strong, ~99.5% turn probability)
    """
    
    def __init__(self, k_actions, w_forward=2.2, w_alternating=1.8, 
                 w_outward=1.0, w_branch_explore=0.16, w_branch_home=10.0):
        self.k = k_actions 
        self.total_actions = self.k + 1
        self.w_forward = w_forward
        self.w_alternating = w_alternating
        self.w_outward = w_outward
        self.w_branch_explore = w_branch_explore
        self.w_branch_home = w_branch_home
        
        # Working memory (1 step)
        self.prev_action = None 
        self.banned_branch = None
        
        # Memory cost: 5 weights + 2 state variables
        self.memory_bits = (5 * 32) + (2 * 32)

    def choose_action(self, state, has_water, at_dead_end):
        """
        Choose action based on local heuristics.
        
        Priority order:
        1. Whisker reflex: If at dead end → reverse
        2. Homing: If carrying water → strong branch bias
        3. Exploration: Standard biases
        """
        
        # Priority 1: Whisker reflex
        if at_dead_end:
            if self.prev_action is not None and self.prev_action < self.k:
                self.banned_branch = self.prev_action
            self.prev_action = self.k
            return self.k
        
        # Priority 2: Homing behavior
        if has_water:
            return self._homing_action()
        
        # Priority 3: Exploration
        return self._exploration_action()
    
    def _homing_action(self):
        """
        Homing strategy: STRONG preference for branches over straight.
        Implements one-shot homing by preferring turns.
        """
        logits = np.zeros(self.total_actions)
        
        # Base forward bias
        for i in range(self.k):
            logits[i] += self.w_forward
        
        # STRONG branch bias
        if self.k == 2:
            logits[1] += self.w_branch_home   # Branch: huge boost
            logits[0] -= self.w_branch_home   # Straight: huge penalty
        else:
            # k-ary: prefer peripheral branches
            for i in range(self.k):
                branch_weight = self.w_branch_home * (i / (self.k - 1))
                logits[i] += branch_weight
        
        # Softmax selection
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        
        action = np.random.choice(np.arange(self.total_actions), p=probs)
        self.prev_action = action
        self.banned_branch = None
        
        return action
    
    def _exploration_action(self):
        """
        Exploration strategy: All standard mouse turning biases.
        """
        logits = np.zeros(self.total_actions) 
        
        # Forward bias (P_SF ≈ 0.88)
        for i in range(self.k):
            logits[i] += self.w_forward
        
        # Subtle branch bias (P_BS ≈ 0.54)
        if self.k == 2:
            logits[1] += self.w_branch_explore
            logits[0] -= self.w_branch_explore
        else:
            for i in range(self.k):
                branch_weight = self.w_branch_explore * (i / (self.k - 1))
                logits[i] += branch_weight
        
        # Context-dependent biases
        if self.prev_action is not None:
            # Alternation bias (P_SA ≈ 0.72)
            if self.prev_action < self.k:
                self.banned_branch = None
                for i in range(self.k):
                    if i != self.prev_action:
                        logits[i] += self.w_alternating
            # Outward bias
            elif self.prev_action == self.k:
                for i in range(self.k):
                    logits[i] += self.w_outward
        
        # Working memory penalty
        if self.banned_branch is not None:
            logits[self.banned_branch] -= 100.0
        
        # Softmax selection
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        
        action = np.random.choice(np.arange(self.total_actions), p=probs)
        self.prev_action = action
        
        return action
# =============================================================================
# Q-LEARNING WITH MEMORY DECAY
# =============================================================================

class QLearningWithDecay(QLearning_agent):
    """
    Q-learning agent with synaptic decay.
    
    Biological motivation: Unused synapses decay over time due to
    lack of protein synthesis and maintenance. This adds realism
    to the memory cost model.
    
    Reference: Harris et al. (2012) - Synaptic energy use and supply
    """
    
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99, 
                 decay_rate=0.001, decay_threshold=100):
        super().__init__(num_states, num_actions, alpha, gamma)
        
        self.decay_rate = decay_rate
        self.decay_threshold = decay_threshold
        
        # Track last access time for each state-action pair
        self.last_access_time = np.zeros((num_states, num_actions))
        self.current_time = 0
        
    def choose_action(self, state):
        """Choose action and apply decay to unused memories."""
        self.current_time += 1
        self._apply_decay()
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """Standard Q-learning update."""
        self.visited_states.add(state)
        self.last_access_time[state, action] = self.current_time
        
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def _apply_decay(self):
        """Apply synaptic decay to unused memories."""
        time_since_access = self.current_time - self.last_access_time
        
        # Decay Q-values not accessed recently
        decay_mask = time_since_access > self.decay_threshold
        self.q_table[decay_mask] *= (1 - self.decay_rate)
        
        # Prune very small Q-values (synaptic elimination)
        self.q_table[np.abs(self.q_table) < 0.01] = 0
    
    def get_active_memory_bits(self):
        """Calculate memory cost based only on non-zero Q-values."""
        return np.count_nonzero(self.q_table) * 32