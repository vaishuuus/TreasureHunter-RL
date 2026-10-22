import numpy as np


class TreasureHunterEnv:
    def __init__(self):
        # Grid size
        self.rows = 6
        self.cols = 6

        # Maximum number of steps allowed in one episode
        self.max_steps = 50

        # Actions:
        # 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
        self.action_map = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

        # Fixed starting position
        self.start_pos = (0, 0)

        # Base grid
        # 0  = empty
        # -1 = wall
        # 2  = treasure
        # -2 = trap
        # 3  = exit

        self.base_grid = np.zeros((self.rows, self.cols), dtype=np.float32)

        # Walls
        walls = [
            (0, 3),
            (1, 1),
            (3, 1),
            (4, 3)
        ]

        for row, col in walls:
            self.base_grid[row, col] = -1

        # Treasure
        self.treasure_pos = (2, 2)
        self.base_grid[self.treasure_pos] = 2

        # Trap
        self.trap_pos = (2, 4)
        self.base_grid[self.trap_pos] = -2

        # Exit
        self.exit_pos = (5, 5)
        self.base_grid[self.exit_pos] = 3

        # Start the environment
        self.reset()

    def reset(self):
        """Reset the environment for a new episode."""

        self.grid = self.base_grid.copy()

        self.agent_pos = self.start_pos
        self.treasure_collected = False
        self.steps = 0
        self.done = False

        return self._get_state()

    def _get_state(self):
        """
        Convert the current environment into a numerical state.

        Grid = 36 values
        Extra value = whether treasure has been collected
        Total state size = 37
        """

        state = self.grid.astype(np.float32).reshape(-1)

        # Put the agent into the state representation
        agent_index = self.agent_pos[0] * self.cols + self.agent_pos[1]
        state[agent_index] = 1.0

        # Add treasure-collected information
        treasure_status = 1.0 if self.treasure_collected else 0.0

        state = np.append(state, treasure_status)

        return state

    def step(self, action):
        """
        Perform one action.

        Returns:
            next_state
            reward
            done
            info
        """

        if self.done:
            return self._get_state(), 0.0, True, {}

        self.steps += 1

        # Get movement direction
        move_row, move_col = self.action_map[action]

        current_row, current_col = self.agent_pos

        new_row = current_row + move_row
        new_col = current_col + move_col

        # Default reward for a normal move
        reward = -0.1

        # Check whether the new position is outside the grid
        if (
            new_row < 0
            or new_row >= self.rows
            or new_col < 0
            or new_col >= self.cols
        ):
            reward = -2.0

        # Check whether the new position is a wall
        elif self.grid[new_row, new_col] == -1:
            reward = -2.0

        else:
            # Move the agent
            self.agent_pos = (new_row, new_col)

            cell = self.grid[new_row, new_col]

            # Treasure
            if cell == 2:
                reward = 10.0
                self.treasure_collected = True

                # Remove treasure from the grid
                self.grid[new_row, new_col] = 0

            # Trap
            elif cell == -2:
                reward = -10.0
                self.done = True

            # Exit
            elif cell == 3:
                if self.treasure_collected:
                    reward = 20.0
                    self.done = True
                else:
                    # Exit reached too early
                    reward = -2.0

        # End episode if maximum steps are reached
        if self.steps >= self.max_steps:
            self.done = True

        next_state = self._get_state()

        info = {
            "treasure_collected": self.treasure_collected,
            "steps": self.steps
        }

        return next_state, reward, self.done, info

    def render(self):
        """Display the current grid in the terminal."""

        symbols = {
            0: "·",
            -1: "█",
            2: "💎",
            -2: "☠",
            3: "🏆"
        }

        print()

        for row in range(self.rows):
            row_text = ""

            for col in range(self.cols):

                if (row, col) == self.agent_pos:
                    row_text += "🤖 "

                else:
                    row_text += symbols[self.grid[row, col]] + " "

            print(row_text)

        print(f"Steps: {self.steps}")
        print(f"Treasure collected: {self.treasure_collected}")
        print(f"Finished: {self.done}")
        print()


if __name__ == "__main__":

    env = TreasureHunterEnv()

    print("TREASURE HUNTER - ENVIRONMENT TEST")
    print("-----------------------------------")

    env.render()

    # Test a valid path:
    # DOWN, DOWN, RIGHT, RIGHT,
    # DOWN, DOWN, DOWN,
    # RIGHT, RIGHT, RIGHT

    actions = [
        1, 1,
        3, 3,
        1, 1, 1,
        3, 3, 3
    ]

    total_reward = 0

    for action in actions:

        state, reward, done, info = env.step(action)

        total_reward += reward

        print(f"Action: {action}")
        print(f"Reward: {reward}")

        env.render()

        if done:
            break

    print("Final Result")
    print("------------")
    print(f"Total reward: {total_reward}")
    print(f"Treasure collected: {env.treasure_collected}")
    print(f"Episode finished: {env.done}")