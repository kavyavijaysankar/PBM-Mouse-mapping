import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LabyrinthEnv(gym.Env):
    def __init__(self, k=2, depth=6, goal_node=None):
        super(LabyrinthEnv, self).__init__()
        
        self.k = k
        self.depth = depth
        # General formula for total nodes in a k-ary tree
        self.num_nodes = (self.k ** (self.depth + 1) - 1) // (self.k - 1)
        
        self.start_node = 0 
        
        # If no goal is provided, pick the last leaf node
        if goal_node is None:
            self.goal_node = self.num_nodes - 1
        else:
            self.goal_node = goal_node 
            
        # STATE SPACE DOUBLED: Half for searching, half for returning with water
        self.observation_space = spaces.Discrete(self.num_nodes * 2)
        
        # ACTION SPACE: 0 to (k-1) are Forward moves. Action k is the Reverse move.
        self.action_space = spaces.Discrete(self.k + 1)
        
        self.current_node = self.start_node
        self.has_water = False
        
    def _get_obs(self):
        # Flattens the state. If it has water, shift the state index up by num_nodes.
        return self.current_node + (self.num_nodes if self.has_water else 0)

    def _get_info(self):
        # Calculate the first leaf node index to determine dead ends
        first_leaf = (self.k ** self.depth - 1) // (self.k - 1)
        at_dead_end = (self.current_node >= first_leaf)
        
        return {
            'node': self.current_node, 
            'has_water': self.has_water, 
            'at_dead_end': at_dead_end
        }
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_node = self.start_node
        self.has_water = False
        return self._get_obs(), self._get_info()

    def step(self, action):
        reward = -0.01  # Metabolic step penalty
        terminated = False
        truncated = False
        
        # --- 1. MOVEMENT LOGIC ---
        if action < self.k:
            # Forward move
            next_node = self.k * self.current_node + action + 1
        elif action == self.k:
            # Reverse move
            next_node = (self.current_node - 1) // self.k
        else:
            raise ValueError("Invalid action")
            
        # Bounds checking (Walls)
        if next_node >= self.num_nodes:
            next_node = self.current_node # Hit a leaf, stay in place
        elif self.current_node == 0 and action == self.k:
            next_node = self.current_node # At entrance, can't reverse
            
        self.current_node = next_node
        
        # --- 2. GOAL LOGIC (The Round Trip) ---
        if self.current_node == self.goal_node and not self.has_water:
            self.has_water = True
            reward = 1.0  # Reward for finding the water
            
        if self.current_node == self.start_node and self.has_water:
            reward += 10.0  # Big reward for successfully returning home
            terminated = True
            
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        phase = "Returning" if self.has_water else "Searching"
        print(f"Phase: {phase} | Node: {self.current_node}")