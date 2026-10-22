from environment import TreasureHunterEnv
from dqn_agent import DQNAgent
import torch

print("=" * 45)
print("TREASURE HUNTER - DQN TEST")
print("=" * 45)

# Create environment and agent
env = TreasureHunterEnv()
agent = DQNAgent(state_size=37, action_size=4)

# Test state
state = env.reset()

print("\nState size:", len(state))
print("Expected size: 37")

# Test action selection
action = agent.select_action(state)

print("Selected action:", action)
print("Expected action range: 0 to 3")

# Test neural network output
state_tensor = torch.tensor(
    state,
    dtype=torch.float32,
    device=agent.device
).unsqueeze(0)

with torch.no_grad():
    q_values = agent.policy_network(state_tensor)

print("\nQ-values:", q_values)
print("Output shape:", tuple(q_values.shape))

# Test replay buffer
next_state, reward, done, info = env.step(action)

agent.remember(
    state,
    action,
    reward,
    next_state,
    done
)

print("\nReplay buffer size:", len(agent.replay_buffer))

# Verify everything
print("\n--- TEST RESULTS ---")

assert len(state) == 37, "Incorrect state size"
print("PASS: State size")

assert 0 <= action <= 3, "Invalid action"
print("PASS: Action selection")

assert tuple(q_values.shape) == (1, 4), "Incorrect output shape"
print("PASS: Four Q-values produced")

assert len(agent.replay_buffer) == 1, "Experience not stored"
print("PASS: Replay buffer")

print("\nAll DQN tests passed!")