"""
Step-by-Step Diagnostic - Trace One Full Episode

This will show exactly where the agent gets stuck.
"""

import sys
sys.path.insert(0, '.')  # Current directory

print("Attempting to import modules...")
try:
    from labyrinth import LabyrinthEnv
    from Agent import HeuristicAgent
    print("✓ Imports successful\n")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

print("="*70)
print("DETAILED EPISODE TRACE")
print("="*70)

# Create environment and agent
env = LabyrinthEnv(k=2, depth=6)
agent = HeuristicAgent(k_actions=2)

print(f"\nEnvironment setup:")
print(f"  Start node: {env.start_node}")
print(f"  Goal node: {env.goal_node}")
print(f"  First leaf: {(2**6 - 1) // (2-1)}")

# Run one episode with detailed logging
obs, info = env.reset(seed=42)
agent.reset()

print(f"\nStarting episode:")
print(f"  Initial node: {info['node']}")
print(f"  Has water: {info['has_water']}")
print(f"  At dead end: {info['at_dead_end']}")

done = False
steps = 0
max_steps = 100  # Shorter limit for debugging

found_water = False
visiting_pattern = []

print(f"\n{'Step':<5} {'Node':<5} {'Water':<6} {'DeadEnd':<8} {'Action':<10} {'Next':<5} {'Reward':<8}")
print("-"*70)

while not done and steps < max_steps:
    # Record state
    current_node = info['node']
    has_water = info['has_water']
    at_dead_end = info['at_dead_end']
    
    # Choose action
    action = agent.choose_action(obs, has_water, at_dead_end)
    
    # Take step
    next_obs, reward, done, truncated, next_info = env.step(action)
    
    # Record pattern
    visiting_pattern.append((current_node, action, next_info['node']))
    
    # Action name
    if action < 2:
        action_name = f"forward_{action}"
    else:
        action_name = "reverse"
    
    # Water emoji
    water_emoji = "💧" if has_water else "  "
    
    # Print step info
    print(f"{steps:<5} {current_node:<5} {water_emoji:<6} {str(at_dead_end):<8} {action_name:<10} {next_info['node']:<5} {reward:<8.2f}")
    
    # Check milestones
    if next_info['node'] == env.goal_node and not has_water:
        print(f"      >>> FOUND WATER at step {steps}!")
        found_water = True
    
    if next_info['node'] == env.start_node and next_info['has_water']:
        print(f"      >>> RETURNED HOME at step {steps}!")
    
    # Update state
    obs = next_obs
    info = next_info
    steps += 1

print("\n" + "="*70)
print("EPISODE SUMMARY")
print("="*70)
print(f"Steps taken: {steps}")
print(f"Episode completed: {done}")
print(f"Found water: {found_water}")
print(f"Final node: {info['node']}")
print(f"Has water: {info['has_water']}")

# Analyze visiting pattern
if not done and steps >= max_steps:
    print(f"\n⚠ Hit step limit!")
    
    # Check for loops
    last_10 = visiting_pattern[-10:]
    nodes_visited = [v[0] for v in last_10]
    actions_taken = [v[1] for v in last_10]
    
    print(f"\nLast 10 steps:")
    for i, (node, action, next_node) in enumerate(last_10):
        act_name = f"fwd_{action}" if action < 2 else "rev"
        print(f"  {node} --{act_name}--> {next_node}")
    
    # Check for oscillation
    if len(set(nodes_visited)) <= 3:
        print(f"\n🐛 STUCK IN LOOP!")
        print(f"   Oscillating between nodes: {set(nodes_visited)}")
    
    # Check if stuck at one node
    if len(set(nodes_visited)) == 1:
        print(f"\n🐛 STUCK AT SINGLE NODE: {nodes_visited[0]}")
        print(f"   Actions keep returning to same node")

# Check agent state
print(f"\nAgent internal state:")
print(f"  prev_action: {agent.prev_action}")
print(f"  banned_branch: {agent.banned_branch}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

if not found_water:
    print("✗ Agent never found water during exploration phase")
    print("  → Issue is in exploration logic, not homing")
elif found_water and not done:
    print("✓ Agent found water")
    print("✗ Agent did not return home")
    print("  → Issue is in homing logic")
elif done:
    print("✓ Episode completed successfully!")
    print(f"  → Heuristic agent works! (took {steps} steps)")