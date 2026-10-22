
import os
import numpy as np

from environment import TreasureHunterEnv
from dqn_agent import DQNAgent


# ==========================================
# 1. LOAD THE TRAINED MODEL
# ==========================================

MODEL_PATH = "models/treasure_hunter_dqn.pth"
NUM_EPISODES = 30

env = TreasureHunterEnv()

initial_state = env.reset()

agent = DQNAgent(
    state_size=len(initial_state),
    action_size=4
)

agent.load(MODEL_PATH)

print("\nTREASURE HUNTER - DQN EVALUATION")
print("=" * 45)
print("Exploration: DISABLED")
print("Evaluation episodes:", NUM_EPISODES)


# ==========================================
# 2. EVALUATE THE AGENT
# ==========================================

episode_rewards = []
episode_steps = []
successes = []
first_episode_actions = []

action_names = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}

for episode in range(1, NUM_EPISODES + 1):

    state = env.reset()
    total_reward = 0.0

    while True:

        # Use learned Q-values; no random exploration
        action = agent.select_action(
            state,
            training=False
        )

        if episode == 1:
            first_episode_actions.append(
                action_names[action]
            )

        next_state, reward, done, info = env.step(action)

        total_reward += reward
        state = next_state

        if done:
            break

    success = (
        env.treasure_collected
        and env.agent_pos == env.exit_pos
    )

    episode_rewards.append(total_reward)
    episode_steps.append(env.steps)
    successes.append(int(success))

    if episode % 10 == 0:
        print(
            f"Episode {episode:2d}/{NUM_EPISODES} | "
            f"Reward: {total_reward:6.2f} | "
            f"Steps: {env.steps:2d} | "
            f"Success: {success}"
        )


# ==========================================
# 3. SUMMARIZE RESULTS
# ==========================================

success_rate = np.mean(successes) * 100
average_reward = np.mean(episode_rewards)
average_steps = np.mean(episode_steps)

print("\n" + "=" * 45)
print("FINAL EVALUATION RESULTS")
print("=" * 45)

print(f"Success rate:       {success_rate:.1f}%")
print(f"Average reward:     {average_reward:.2f}")
print(f"Average steps:      {average_steps:.2f}")
print(f"Successful runs:    {sum(successes)}/{NUM_EPISODES}")

print("\nFirst episode's actions:")
print(" -> ".join(first_episode_actions))


# ==========================================
# 4. SAVE THE RESULTS
# ==========================================

os.makedirs("results", exist_ok=True)

with open(
    "results/evaluation_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("Treasure Hunter - DQN Evaluation\n")
    file.write("=" * 40 + "\n")
    file.write(f"Episodes: {NUM_EPISODES}\n")
    file.write(f"Exploration: Disabled\n")
    file.write(f"Success rate: {success_rate:.1f}%\n")
    file.write(f"Average reward: {average_reward:.2f}\n")
    file.write(f"Average steps: {average_steps:.2f}\n")
    file.write(
        f"Successful runs: {sum(successes)}/{NUM_EPISODES}\n"
    )

print("\nSaved to results/evaluation_results.txt")
