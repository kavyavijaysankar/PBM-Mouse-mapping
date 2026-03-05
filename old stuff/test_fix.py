"""
Quick Test: Verify Heuristic Agent Fix
"""

import sys
sys.path.insert(0, '/mnt/user-data/outputs')

from labyrinth import LabyrinthEnv
from Agent import HeuristicAgent

print("Testing heuristic agent fix...")

env = LabyrinthEnv(k=2, depth=6)
agent = HeuristicAgent(k_actions=2)

# Test 3 episodes
episode_steps = []

for episode in range(3):
    obs, info = env.reset(seed=episode)
    agent.reset()  # CRITICAL: Reset agent state
    
    done = False
    steps = 0
    
    while not done and steps < 500:
        action = agent.choose_action(obs, info['has_water'], info['at_dead_end'])
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
    
    episode_steps.append(steps)
    status = "✓ Completed" if done else "✗ Hit limit"
    print(f"Episode {episode+1}: {steps:3d} steps  {status}")

print(f"\nAverage: {sum(episode_steps)/len(episode_steps):.1f} steps")

if all(s < 500 for s in episode_steps):
    print("\n✓ FIX SUCCESSFUL! All episodes completed.")
    if sum(episode_steps)/len(episode_steps) < 100:
        print("✓ Performance looks good (average < 100 steps)")
else:
    print("\n✗ Still hitting 500-step limit. Further debugging needed.")