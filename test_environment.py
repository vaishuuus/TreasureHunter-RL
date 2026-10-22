from environment import TreasureHunterEnv


def run_test(name, actions):
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    env = TreasureHunterEnv()

    print("Initial position:", env.agent_pos)

    total_reward = 0

    for action in actions:
        state, reward, done, info = env.step(action)

        total_reward += reward

        print(
            f"Action: {action} | "
            f"Position: {env.agent_pos} | "
            f"Reward: {reward} | "
            f"Done: {done}"
        )

        if done:
            break

    print("\nFinal position:", env.agent_pos)
    print("Total reward:", total_reward)
    print("Treasure collected:", env.treasure_collected)
    print("Episode finished:", env.done)


# TEST 1
# Try to move UP from the starting position.
# Agent should stay at (0, 0).
run_test(
    "TEST 1 - Outside Grid",
    [0]
)


# TEST 2
# Move RIGHT twice to (0, 2),
# then try RIGHT again into a wall at (0, 3).
run_test(
    "TEST 2 - Wall Collision",
    [3, 3, 3]
)


# TEST 3
# Move to the trap at (2, 4).
run_test(
    "TEST 3 - Trap",
    [1, 1, 3, 3, 3, 3]
)


# TEST 4
# Reach the exit without collecting the treasure.
run_test(
    "TEST 4 - Exit Without Treasure",
    [1, 1, 1, 1, 1, 3, 3, 3, 3, 3]
)


# TEST 5
# Keep taking invalid actions until maximum steps are reached.
run_test(
    "TEST 5 - Maximum Steps",
    [0] * 50
)