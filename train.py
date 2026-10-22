
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch

from environment import TreasureHunterEnv
from dqn_agent import DQNAgent


# ==========================================
# 1. SETTINGS
# ==========================================

NUM_EPISODES = 1000
PRINT_EVERY = 50

# Make results more reproducible
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ==========================================
# 2. PREPARE PROJECT
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

env = TreasureHunterEnv()

initial_state = env.reset()

agent = DQNAgent(
    state_size=len(initial_state),
    action_size=4
)

episode_rewards = []
success_history = []
loss_history = []


# ==========================================
# 3. TRAIN THE AGENT
# ==========================================

print("\nStarting DQN training...")
print("Total episodes:", NUM_EPISODES)

for episode in range(1, NUM_EPISODES + 1):

    state = env.reset()
    total_reward = 0.0

    while True:

        # Choose an action
        action = agent.select_action(
            state,
            training=True
        )

        # Perform the action
        next_state, reward, done, info = env.step(action)

        # Remember the experience
        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        # Learn from past experiences
        loss = agent.train_step()

        if loss is not None:
            loss_history.append(loss)

        # Move to the next state
        state = next_state
        total_reward += reward

        if done:
            break

    # Was the objective successfully completed?
    success = (
        env.treasure_collected
        and env.agent_pos == env.exit_pos
    )

    episode_rewards.append(total_reward)
    success_history.append(int(success))

    # Display progress every 50 episodes
    if episode % PRINT_EVERY == 0:

        recent_rewards = episode_rewards[-PRINT_EVERY:]
        recent_successes = success_history[-PRINT_EVERY:]

        average_reward = np.mean(recent_rewards)
        success_rate = 100 * np.mean(recent_successes)

        print(
            f"Episode {episode:4d}/{NUM_EPISODES} | "
            f"Avg Reward: {average_reward:7.2f} | "
            f"Success Rate: {success_rate:5.1f}% | "
            f"Epsilon: {agent.epsilon:.3f}"
        )


# ==========================================
# 4. SAVE TRAINED MODEL AND DATA
# ==========================================

model_path = "models/treasure_hunter_dqn.pth"

agent.save(model_path)

np.savez(
    "results/training_data.npz",
    rewards=np.array(episode_rewards),
    successes=np.array(success_history),
    losses=np.array(loss_history)
)

print("\nTraining finished.")
print("Model saved to:", model_path)
print("Training data saved to results/training_data.npz")


# ==========================================
# 5. PLOT REWARD CURVE
# ==========================================

window = 50

moving_rewards = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(10, 5))

plt.plot(
    episode_rewards,
    alpha=0.3,
    label="Episode reward"
)

plt.plot(
    np.arange(window - 1, NUM_EPISODES),
    moving_rewards,
    label="50-episode average"
)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Treasure Hunter - DQN Training Rewards")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("results/reward_curve.png", dpi=150)
plt.close()


# ==========================================
# 6. PLOT SUCCESS RATE
# ==========================================

moving_success = np.convolve(
    success_history,
    np.ones(window) / window,
    mode="valid"
) * 100

plt.figure(figsize=(10, 5))

plt.plot(
    np.arange(window - 1, NUM_EPISODES),
    moving_success
)

plt.xlabel("Episode")
plt.ylabel("Success Rate (%)")
plt.title("Treasure Hunter - Success Rate")
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("results/success_rate.png", dpi=150)
plt.close()


# ==========================================
# 7. PLOT TRAINING LOSS
# ==========================================

if loss_history:

    plt.figure(figsize=(10, 5))

    plt.plot(loss_history)

    plt.xlabel("Training Update")
    plt.ylabel("Loss")
    plt.title("Treasure Hunter - DQN Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("results/training_loss.png", dpi=150)
    plt.close()


print("\nGraphs saved in the results folder:")
print("- reward_curve.png")
print("- success_rate.png")
print("- training_loss.png")

print("\nDAY 5 TRAINING SCRIPT COMPLETED")
