"""
Memory Hierarchy for Navigation Agents

This module implements three agents with increasing memory capacity:
- Level 0: Pure reactive (no spatial memory)
- Level 1: Minimal memory (visited nodes)
- Level 2: Full memory (state-action values)
"""

import numpy as np


# =============================================================================
# LEVEL 0: PURE REACTIVE AGENT (No Spatial Memory)
# =============================================================================

class ReactiveAgent:
    """
    Pure stimulus-response agent with NO spatial memory.
    
    Memory: ~224 bits (7 parameters)
    - Only remembers: last action, one banned branch, escape counter
    - Cannot learn locations
    - Cannot remember visited areas
    - Purely reactive to local cues
    
    This represents the absolute minimum for navigation:
    obstacle avoidance and local biases only.
    """
    
    def __init__(self, k_actions, w_forward=2.2, w_alternating=1.8, 
                 w_outward=1.0, w_branch_explore=0.16):
        self.k = k_actions
        self.total_actions = k_actions + 1
        self.w_forward = w_forward
        self.w_alternating = w_alternating
        self.w_outward = w_outward
        self.w_branch_explore = w_branch_explore
        
        # Minimal state (no spatial memory)
        self.prev_action = None
        self.banned_branch = None
        self.escape_mode = 0
        
        # Memory cost: 4 weights + 3 state variables
        self.memory_bits = (4 * 32) + (3 * 32)
    
    def reset(self):
        self.prev_action = None
        self.banned_branch = None
        self.escape_mode = 0
    
    def choose_action(self, state, has_water, at_dead_end):
        """Pure reactive navigation with no spatial memory."""
        
        # Priority 1: Dead end reflex
        if at_dead_end:
            if self.prev_action is not None and self.prev_action < self.k:
                self.banned_branch = self.prev_action
            self.escape_mode = 2
            self.prev_action = self.k
            return self.k
        
        # Priority 2: Escape mode (short-term)
        if self.escape_mode > 0:
            self.escape_mode -= 1
            self.prev_action = self.k
            return self.k
        
        # Priority 3: Homing (when has water)
        if has_water:
            self.prev_action = self.k
            return self.k
        
        # Priority 4: Exploration with local biases
        logits = np.zeros(self.total_actions)
        
        # Forward bias
        for i in range(self.k):
            logits[i] += self.w_forward
        
        # Branch bias
        if self.k == 2:
            logits[1] += self.w_branch_explore
            logits[0] -= self.w_branch_explore
        
        # Alternation bias
        if self.prev_action is not None:
            if self.prev_action < self.k:
                for i in range(self.k):
                    if i != self.prev_action:
                        logits[i] += self.w_alternating
            elif self.prev_action == self.k:
                for i in range(self.k):
                    logits[i] += self.w_outward
        
        # Working memory penalty (one branch only)
        if self.banned_branch is not None:
            logits[self.banned_branch] -= 100.0
        
        # Sample action
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        action = np.random.choice(np.arange(self.total_actions), p=probs)
        
        # Update state
        if action < self.k:
            self.banned_branch = None
            self.escape_mode = 0
        
        self.prev_action = action
        return action


# =============================================================================
# LEVEL 1: MINIMAL MEMORY AGENT (Visited Nodes)
# =============================================================================

class MinimalMemoryAgent(ReactiveAgent):
    """
    Agent with minimal spatial memory: remembers visited nodes.
    
    Memory: ~4,000-8,000 bits (visited node set)
    - Remembers which areas have been explored
    - Enables systematic coverage
    - Still uses local heuristics
    - No state-action learning
    
    This represents simple spatial familiarity:
    "I've been here before" without detailed path knowledge.
    
    Can configure memory capacity to test minimum required.
    """
    
    def __init__(self, k_actions, max_memory_nodes=None, **kwargs):
        super().__init__(k_actions, **kwargs)
        
        self.max_memory_nodes = max_memory_nodes  # None = unlimited
        self.visited_nodes = []  # Use list to maintain order for FIFO
        self.visit_counts = {}
        
        # Memory cost: base + visited nodes
        if max_memory_nodes is None:
            # Full memory: assume ~127 nodes × 32 bits
            memory_nodes = 127
        else:
            memory_nodes = max_memory_nodes
        
        self.memory_bits = (4 * 32) + (3 * 32) + (memory_nodes * 32)
    
    def reset(self):
        super().reset()
        # Keep visited nodes across episodes (long-term memory)
    
    def reset_spatial_memory(self):
        """Clear spatial memory (for new environment)."""
        self.visited_nodes = []
        self.visit_counts = {}
    
    def _update_memory(self, node):
        """Update visited nodes with memory limit."""
        # Add to visit counts
        self.visit_counts[node] = self.visit_counts.get(node, 0) + 1
        
        # Track visit order
        if node not in self.visited_nodes:
            self.visited_nodes.append(node)
            
            # Enforce memory limit (FIFO - forget oldest)
            if self.max_memory_nodes and len(self.visited_nodes) > self.max_memory_nodes:
                oldest = self.visited_nodes.pop(0)
                # Reduce visit count (but don't fully forget)
                if oldest in self.visit_counts:
                    self.visit_counts[oldest] = max(0, self.visit_counts[oldest] - 1)
                    if self.visit_counts[oldest] == 0:
                        del self.visit_counts[oldest]
    
    def choose_action(self, state, has_water, at_dead_end):
        """Navigation with spatial familiarity."""
        
        # Extract current node from doubled state space
        current_node = state % 127  # Assumes 127 nodes
        
        # Update visit tracking
        self._update_memory(current_node)
        
        # Use parent class logic for priorities 1-3
        if at_dead_end:
            if self.prev_action is not None and self.prev_action < self.k:
                self.banned_branch = self.prev_action
            self.escape_mode = 2
            self.prev_action = self.k
            return self.k
        
        if self.escape_mode > 0:
            self.escape_mode -= 1
            self.prev_action = self.k
            return self.k
        
        if has_water:
            self.prev_action = self.k
            return self.k
        
        # Priority 4: Exploration with novelty preference
        logits = np.zeros(self.total_actions)
        
        # Forward bias
        for i in range(self.k):
            logits[i] += self.w_forward
        
        # Branch bias
        if self.k == 2:
            logits[1] += self.w_branch_explore
            logits[0] -= self.w_branch_explore
        
        # Novelty bonus: prefer less-visited areas (GENTLE penalty)
        for i in range(self.k):
            next_node = self.k * current_node + i + 1
            if next_node < 127:  # Valid node
                visit_count = self.visit_counts.get(next_node, 0)
                # GENTLE penalty - don't make it impossible to revisit
                # Use logarithmic penalty to avoid over-penalizing
                if visit_count > 0:
                    logits[i] -= 0.5 * np.log(visit_count + 1)
        
        # Alternation bias
        if self.prev_action is not None:
            if self.prev_action < self.k:
                for i in range(self.k):
                    if i != self.prev_action:
                        logits[i] += self.w_alternating
            elif self.prev_action == self.k:
                for i in range(self.k):
                    logits[i] += self.w_outward
        
        # Working memory penalty
        if self.banned_branch is not None:
            logits[self.banned_branch] -= 100.0
        
        # Sample action
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        action = np.random.choice(np.arange(self.total_actions), p=probs)
        
        # Update state
        if action < self.k:
            self.banned_branch = None
            self.escape_mode = 0
        
        self.prev_action = action
        return action


# =============================================================================
# LEVEL 2: FULL MEMORY AGENT (Q-Learning)
# =============================================================================

class FullMemoryAgent:
    """
    Agent with full state-action memory (Q-table).
    
    Memory: ~24,384 bits (254 states × 3 actions × 32 bits)
    - Remembers value of every state-action pair
    - Learns optimal policy
    - Can plan paths
    - Requires extensive training
    
    This represents a complete cognitive map:
    detailed knowledge of spatial structure and optimal paths.
    """
    
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.05
        
        self.q_table = np.zeros((num_states, num_actions))
        self.visited_states = set()
        
        # Memory cost: full Q-table
        self.memory_bits = num_states * num_actions * 32
    
    def choose_action(self, state):
        """Epsilon-greedy with learned Q-values."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """Q-learning update."""
        self.visited_states.add(state)
        
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# =============================================================================
# USAGE SUMMARY
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MEMORY HIERARCHY FOR NAVIGATION")
    print("="*70)
    
    print("\nLevel 0: Reactive Agent")
    reactive = ReactiveAgent(k_actions=2)
    print(f"  Memory: {reactive.memory_bits:,} bits")
    print(f"  Capabilities: Local reflexes only")
    print(f"  Limitation: Cannot learn locations")
    
    print("\nLevel 1: Minimal Memory Agent")
    minimal = MinimalMemoryAgent(k_actions=2)
    print(f"  Memory: {minimal.memory_bits:,} bits")
    print(f"  Capabilities: + Remembers visited areas")
    print(f"  Limitation: No path planning")
    
    print("\nLevel 2: Full Memory Agent (Q-learning)")
    full = FullMemoryAgent(num_states=254, num_actions=3)
    print(f"  Memory: {full.memory_bits:,} bits")
    print(f"  Capabilities: + Learns optimal paths")
    print(f"  Limitation: Requires extensive training")
    
    print("\n" + "="*70)
    print("MEMORY SCALING")
    print("="*70)
    print(f"Level 0 → Level 1: {minimal.memory_bits / reactive.memory_bits:.1f}x increase")
    print(f"Level 1 → Level 2: {full.memory_bits / minimal.memory_bits:.1f}x increase")
    print(f"Level 0 → Level 2: {full.memory_bits / reactive.memory_bits:.1f}x increase")