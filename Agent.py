import numpy as np

class QLearning_agent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.05
        
        self.q_table = np.zeros((num_states, num_actions))
        
        # COST CALCULATION: states * actions * 32-bit floats
        self.memory_bits = num_states * num_actions * 32

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


class HeuristicAgent:
    def __init__(self, k_actions, w_forward=2.0, w_alternating=1.5, w_outward=1.0, w_direction=0.1):
        self.k = k_actions 
        self.total_actions = self.k + 1 
        self.w_forward = w_forward
        self.w_alternating = w_alternating
        self.w_outward = w_outward
        self.w_direction = w_direction
        
        self.prev_action = None 
        self.banned_branch = None # <--- NEW: 1-Step Working Memory
        
        # COST CALCULATION: 4 weights * 32-bits + 3 memory variables
        self.memory_bits = (4 * 32) + 3

    def choose_action(self, state, has_water, at_dead_end):
        # 1. The Home Reflex
        if has_water:
            self.prev_action = self.k
            self.banned_branch = None
            return self.k 
            
        # 2. The Whisker Reflex (Bounce)
        if at_dead_end:
            # Remember the specific branch we hit the wall on!
            if self.prev_action is not None and self.prev_action < self.k:
                self.banned_branch = self.prev_action
            self.prev_action = self.k
            return self.k

        logits = np.zeros(self.total_actions) 
        
        # 3. Forward Biases
        for i in range(self.k):
            logits[i] += self.w_forward
            if i == 0:
                logits[i] += self.w_direction
            else:
                logits[i] -= self.w_direction
        
        # 4. Alternating & Outward Biases
        if self.prev_action is not None:
            if self.prev_action < self.k:
                # We successfully moved forward. Clear the banned branch.
                self.banned_branch = None 
                for i in range(self.k):
                    if i != self.prev_action:
                        logits[i] += self.w_alternating
            elif self.prev_action == self.k:
                # We just reversed. Bias going into a novel forward branch.
                for i in range(self.k):
                    logits[i] += self.w_outward
        
        # Apply a penalty to the branch just visited
        if self.banned_branch is not None:
            logits[self.banned_branch] -= 100.0 
            
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        
        action = np.random.choice(np.arange(self.total_actions), p=probs)
        self.prev_action = action
        return action