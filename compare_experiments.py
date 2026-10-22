
import numpy as np
import matplotlib.pyplot as plt


# Load both experiment results
baseline = np.load("results/training_data_decay_09999.npz")
faster = np.load("results/training_data_decay_09995.npz")

baseline_rewards = baseline["rewards"]
faster_rewards = faster["rewards"]

baseline_success = baseline["successes"]
faster_success = faster["successes"]

WINDOW = 50


# ==========================================
# 1. PRINT SUMMARY METRICS
# ==========================================

def summarize(name, rewards, successes):

    print("\n" + "=" * 48)
    print(name)
    print("=" * 48)

    print(
        "First 100 episodes - average reward:",
        round(float(np.mean(rewards[:100])), 2)
    )

    print(
        "Last 100 episodes - average reward:",
        round(float(np.mean(rewards[-100:])), 2)
    )

    print(
        "First 100 episodes - success rate:",
        round(float(np.mean(successes[:100]) * 100), 2),
        "%"
    )

    print(
        "Last 100 episodes - success rate:",
        round(float(np.mean(successes[-100:]) * 100), 2),
        "%"
    )


summarize(
    "Baseline: epsilon decay 0.9999",
    baseline_rewards,
    baseline_success
)

summarize(
    "Faster decay: epsilon decay 0.9995",
    faster_rewards,
    faster_success
)


# ==========================================
# 2. REWARD COMPARISON
# ==========================================

def moving_average(values, window):

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid"
    )


baseline_avg = moving_average(baseline_rewards, WINDOW)
faster_avg = moving_average(faster_rewards, WINDOW)

baseline_x = np.arange(WINDOW - 1, len(baseline_rewards))
faster_x = np.arange(WINDOW - 1, len(faster_rewards))

plt.figure(figsize=(10, 5))

plt.plot(
    baseline_x,
    baseline_avg,
    label="Decay = 0.9999"
)

plt.plot(
    faster_x,
    faster_avg,
    label="Decay = 0.9995"
)

plt.xlabel("Episode")
plt.ylabel("Average Reward (50 episodes)")
plt.title("Effect of Epsilon Decay on DQN Training")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/epsilon_reward_comparison.png",
    dpi=150
)

plt.close()


# ==========================================
# 3. SUCCESS RATE COMPARISON
# ==========================================

baseline_success_avg = moving_average(
    baseline_success,
    WINDOW
) * 100

faster_success_avg = moving_average(
    faster_success,
    WINDOW
) * 100

plt.figure(figsize=(10, 5))

plt.plot(
    baseline_x,
    baseline_success_avg,
    label="Decay = 0.9999"
)

plt.plot(
    faster_x,
    faster_success_avg,
    label="Decay = 0.9995"
)

plt.xlabel("Episode")
plt.ylabel("Success Rate (%)")
plt.title("Effect of Epsilon Decay on Success Rate")
plt.ylim(0, 100)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/epsilon_success_comparison.png",
    dpi=150
)

plt.close()


print("\nComparison complete!")
print("Saved: results/epsilon_reward_comparison.png")
print("Saved: results/epsilon_success_comparison.png")
