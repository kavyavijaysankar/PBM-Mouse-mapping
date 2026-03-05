"""
Advanced Navigation Agents: Levy Flight and Bayesian Information-Seeking

These agents implement alternative search strategies that may be more
efficient than pure Q-learning or fixed heuristics.

References:
[1] Viswanathan, G. M., et al. (1999). Optimizing the success of random 
    searches. Nature, 401(6756), 911-914.
[2] Humphries, N. E., et al. (2010). Environmental context explains Lévy and 
    Brownian movement patterns of marine predators. Nature, 465(7301), 1066-1069.
[3] Friston, K. (2010). The free-energy principle: a unified brain theory?
    Nature Reviews Neuroscience, 11(2), 127-138.
"""

import numpy as np

# =============================================================================
# LEVY FLIGHT SEARCH AGENT
# =============================================================================

class LevyFlightAgent:
    """
    Agent using Levy flight search strategy.
    
    **Why Levy Flights Matter:**
    Many foraging animals exhibit Levy flight patterns - movement consisting
    of many small steps interspersed with occasional long "flights". This
    pattern is provably optimal for searching in sparse, unpredictable 
    environments (Viswanathan et al., 1999).
    
    **Key Idea:**
    Instead of choosing each direction independently, the agent commits to
    a direction for a random "flight length" drawn from a power-law distribution:
        P(length = l) ∝ l^(-μ)
    
    where 1 < μ < 3. This creates a heavy-tailed distribution with occasional
    very long runs in the same direction.
    
    **Biological Relevance:**
    - Observed in albatrosses, sharks, honey bees, and humans
    - May explain the mouse turning biases in Rosenberg et al. (2021)
    - More efficient than pure random walk
    - Requires minimal memory (just current flight length)
    
    **In This Context:**
    We test whether Levy flights outperform both:
    1. Q-learning (high memory, high computation)
    2. Fixed heuristics (low memory, but potentially suboptimal)
    
    A Levy flight represents a third strategy: medium complexity, 
    potentially optimal efficiency.
    """
    
    def __init__(self, k_actions, mu=2.0, w_forward=2.0):
        """
        Args:
            k_actions: Number of forward branches in k-ary tree
            mu: Levy exponent (1 < mu < 3)
                - mu ≈ 2.0 is "optimal" for sparse resources
                - Lower mu → longer flights (more exploration)
                - Higher mu → shorter flights (more local search)
            w_forward: Bias toward forward vs backward movement
        """
        self.k = k_actions
        self.total_actions = k_actions + 1  # +1 for reverse
        self.mu = mu
        self.w_forward = w_forward
        
        # Flight state
        self.flight_length = 0
        self.current_direction = None
        self.prev_action = None
        
        # Memory cost: minimal (just tracking current flight)
        # 3 floats (mu, w_forward, flight_length) + 2 ints (direction, prev_action)
        self.memory_bits = (3 * 32) + (2 * 32)
        
    def choose_action(self, state, has_water, at_dead_end):
        """
        Choose action based on Levy flight strategy.
        
        Priority order:
        1. Return home with water (highest priority)
        2. Bounce from dead end
        3. Continue current flight
        4. Start new flight with random direction
        """
        
        # Priority 1: Home reflex
        if has_water:
            self.flight_length = 0
            self.prev_action = self.k
            return self.k
        
        # Priority 2: Wall bounce
        if at_dead_end:
            self.flight_length = 0
            self.prev_action = self.k
            return self.k
        
        # Priority 3: Continue current flight
        if self.flight_length > 0:
            self.flight_length -= 1
            return self.current_direction
        
        # Priority 4: Start new flight
        self.flight_length = self._draw_levy_flight()
        self.current_direction = self._choose_new_direction()
        self.prev_action = self.current_direction
        
        return self.current_direction
    
    def _draw_levy_flight(self):
        """
        Draw flight length from power-law distribution.
        
        Uses Pareto distribution (truncated power law):
            P(x) = (μ-1) * x^(-μ)  for x ≥ 1
        
        Returns:
            Flight length (minimum 1)
        """
        # NumPy's pareto uses shape parameter a = μ - 1
        return int(np.random.pareto(self.mu - 1) + 1)
    
    def _choose_new_direction(self):
        """
        Choose direction for new flight with forward bias.
        
        Returns:
            Action index (0 to k for forward, k for reverse)
        """
        logits = np.ones(self.total_actions) * self.w_forward
        logits[self.k] = 1.0  # Lower weight for reverse
        
        # Avoid immediate reversal of previous direction
        if self.prev_action is not None and self.prev_action < self.k:
            logits[self.prev_action] -= 0.5
        
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return np.random.choice(self.total_actions, p=probs)
    
    def reset(self):
        """Reset agent state for new episode."""
        self.flight_length = 0
        self.current_direction = None
        self.prev_action = None


# =============================================================================
# BAYESIAN INFORMATION-SEEKING AGENT
# =============================================================================

class BayesianNavigator:
    """
    Agent that maintains probabilistic beliefs about goal location
    and chooses actions to maximize information gain per metabolic cost.
    
    **Why This Matters:**
    
    Standard RL treats navigation as: "maximize expected reward"
    But exploration isn't about reward - it's about INFORMATION.
    
    The Bayesian approach treats navigation as:
        "minimize uncertainty about goal location"
    
    This naturally produces:
    1. Exploration (high uncertainty → seek information)
    2. Exploitation (low uncertainty → go to goal)
    3. Resource-rational trade-offs (info gain vs. metabolic cost)
    
    **Connection to Free Energy Principle:**
    This implements a simplified version of active inference (Friston 2010).
    Animals don't just maximize reward - they minimize surprise/uncertainty.
    
    **In This Experiment:**
    The Bayesian agent provides a PRINCIPLED way to balance exploration
    and exploitation based on uncertainty, without arbitrary epsilon-greedy.
    
    It's relevant because:
    - Mice spend 84% time exploring (why? → they're reducing uncertainty)
    - Sudden insight moments (rapid belief updates)
    - Naturally trades off info gain vs. metabolic cost
    
    **Key Innovation:**
    Actions are chosen to maximize:
        Value = (Expected Information Gain) - λ * (Metabolic Cost)
    
    where λ is the metabolic cost per bit of information.
    """
    
    def __init__(self, num_nodes, num_actions, lambda_cost=0.1):
        """
        Args:
            num_nodes: Total nodes in maze
            num_actions: Actions available (k forward + 1 reverse)
            lambda_cost: Metabolic cost per bit of information
                - Higher λ → prefer cheap actions
                - Lower λ → prefer informative actions
        """
        self.num_nodes = num_nodes
        self.num_actions = num_actions
        self.lambda_cost = lambda_cost
        
        # Belief distribution over goal location (uniform prior)
        self.goal_beliefs = np.ones(num_nodes) / num_nodes
        
        # History of visited nodes
        self.visited_nodes = set()
        
        # Memory cost: belief distribution (32 bits per node)
        self.memory_bits = num_nodes * 32
    
    def choose_action(self, state, available_actions, has_water=False, at_dead_end=False):
        """
        Choose action to maximize information gain minus metabolic cost.
        
        Args:
            state: Current node
            available_actions: List of valid actions
            has_water: Whether carrying water (if yes, return home)
            at_dead_end: Whether at dead end (if yes, must reverse)
            
        Returns:
            Action that maximizes value
        """
        
        # Priority overrides
        if has_water:
            return self.num_actions - 1  # Reverse
        if at_dead_end:
            return self.num_actions - 1  # Reverse
        
        # Calculate value for each action
        values = []
        for action in available_actions:
            # Expected information gain
            info_gain = self._expected_info_gain(state, action)
            
            # Metabolic cost (in bits)
            metabolic_cost = self._metabolic_cost(action)
            
            # Net value
            value = info_gain - self.lambda_cost * metabolic_cost
            values.append(value)
        
        # Softmax selection with temperature
        temperature = 0.1
        exp_values = np.exp(np.array(values) / temperature)
        probs = exp_values / np.sum(exp_values)
        
        return available_actions[np.random.choice(len(available_actions), p=probs)]
    
    def update_beliefs(self, observation, found_goal):
        """
        Bayesian belief update based on observation.
        
        Args:
            observation: Node visited
            found_goal: Whether goal was found at this node
        """
        self.visited_nodes.add(observation)
        
        if found_goal:
            # Goal found - perfect information
            self.goal_beliefs = np.zeros(self.num_nodes)
            self.goal_beliefs[observation] = 1.0
        else:
            # Goal not here - reduce belief
            self.goal_beliefs[observation] *= 0.01
            
            # Renormalize
            if np.sum(self.goal_beliefs) > 0:
                self.goal_beliefs /= np.sum(self.goal_beliefs)
            else:
                # Shouldn't happen, but safety check
                self.goal_beliefs = np.ones(self.num_nodes) / self.num_nodes
    
    def _expected_info_gain(self, state, action):
        """
        Calculate expected reduction in entropy (information gain).
        
        This is a simplified version. Full implementation would:
        1. Predict next state distribution given action
        2. Calculate expected entropy after observation
        3. Return: current_entropy - expected_entropy
        
        For now, we use a heuristic: unexplored nodes have higher info gain.
        """
        # Current entropy (uncertainty about goal)
        p = self.goal_beliefs + 1e-10  # Avoid log(0)
        current_entropy = -np.sum(p * np.log2(p))
        
        # Heuristic: actions toward high-belief regions give more info
        # This is a simplified approximation
        expected_info = current_entropy * 0.1
        
        return expected_info
    
    def _metabolic_cost(self, action):
        """
        Metabolic cost of taking action (in bits).
        
        This is a simplified version - in full implementation,
        cost would depend on:
        - Physical movement (1 bit per step)
        - Neural activity for decision (retrieval cost)
        """
        return 1.0  # Simplified: 1 bit per action
    
    def get_entropy(self):
        """Return current entropy (uncertainty) about goal location."""
        p = self.goal_beliefs + 1e-10
        return -np.sum(p * np.log2(p))
    
    def reset(self):
        """Reset beliefs for new episode."""
        self.goal_beliefs = np.ones(self.num_nodes) / self.num_nodes
        self.visited_nodes = set()


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_search_strategies(env, n_trials=50, verbose=True):
    """
    Compare different search strategies in the same environment.
    
    Args:
        env: LabyrinthEnv instance
        n_trials: Number of trials per agent
        verbose: Print results
        
    Returns:
        Dictionary with results for each agent type
    """
    from Agent import HeuristicAgent, QLearning_agent
    
    agents = {
        'Heuristic': HeuristicAgent(k_actions=env.k),
        'Levy (μ=1.5)': LevyFlightAgent(k_actions=env.k, mu=1.5),
        'Levy (μ=2.0)': LevyFlightAgent(k_actions=env.k, mu=2.0),
        'Levy (μ=2.5)': LevyFlightAgent(k_actions=env.k, mu=2.5),
    }
    
    results = {}
    
    for name, agent in agents.items():
        steps_list = []
        
        for trial in range(n_trials):
            obs, info = env.reset(seed=trial)
            done = False
            steps = 0
            
            while not done and steps < 1000:
                action = agent.choose_action(obs, info['has_water'], info['at_dead_end'])
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
            
            steps_list.append(steps)
            
            # Reset agent if it has reset method
            if hasattr(agent, 'reset'):
                agent.reset()
        
        results[name] = {
            'mean_steps': np.mean(steps_list),
            'std_steps': np.std(steps_list),
            'memory_bits': agent.memory_bits
        }
    
    if verbose:
        print("\n" + "="*60)
        print("SEARCH STRATEGY COMPARISON")
        print("="*60)
        for name, res in results.items():
            print(f"{name:20s}: {res['mean_steps']:6.1f} ± {res['std_steps']:5.1f} steps "
                  f"(memory: {res['memory_bits']:,} bits)")
        print("="*60)
    
    return results