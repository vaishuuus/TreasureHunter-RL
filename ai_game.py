
import os
import pygame

from environment import TreasureHunterEnv
from dqn_agent import DQNAgent


# ==========================================
# 1. SETTINGS
# ==========================================

pygame.init()

TILE_SIZE = 90
INFO_HEIGHT = 155
GRID_SIZE = 6

WINDOW_WIDTH = TILE_SIZE * GRID_SIZE
WINDOW_HEIGHT = TILE_SIZE * GRID_SIZE + INFO_HEIGHT

# One move every 450 milliseconds
MOVE_DELAY = 500

screen = pygame.display.set_mode(
    (WINDOW_WIDTH, WINDOW_HEIGHT)
)

pygame.display.set_caption(
    "Treasure Hunter - DQN AI"
)

clock = pygame.time.Clock()

title_font = pygame.font.Font(None, 34)
info_font = pygame.font.Font(None, 25)
small_font = pygame.font.Font(None, 21)


# ==========================================
# 2. COLORS
# ==========================================

BACKGROUND = (235, 235, 235)
EMPTY_COLOR = (245, 245, 245)
GRID_COLOR = (185, 185, 185)

WALL_COLOR = (45, 45, 45)
TREASURE_COLOR = (245, 190, 40)
TRAP_COLOR = (220, 70, 70)
EXIT_COLOR = (70, 180, 100)
AGENT_COLOR = (70, 120, 220)

TEXT_COLOR = (30, 30, 30)
WHITE = (255, 255, 255)


# ==========================================
# 3. LOAD ENVIRONMENT AND TRAINED AI
# ==========================================

env = TreasureHunterEnv()

state = env.reset()

agent = DQNAgent(
    state_size=len(state),
    action_size=4
)

model_path = "models/treasure_hunter_dqn.pth"

if not os.path.exists(model_path):
    pygame.quit()
    raise FileNotFoundError(
        "Trained model not found. Run python train.py first."
    )

agent.load(model_path)

# Disable random exploration during gameplay.
# The AI will choose actions using its learned Q-values.
agent.epsilon = 0.0

print("Trained DQN loaded successfully.")
print("AI gameplay started.")


# ==========================================
# 4. GAME VARIABLES
# ==========================================

last_reward = 0.0
total_reward = 0.0

paused = False
running = True

last_move_time = pygame.time.get_ticks()


# ==========================================
# 5. DRAW THE GRID AND HUNTER
# ==========================================

def draw_grid():

    screen.fill(BACKGROUND)

    for row in range(env.rows):
        for col in range(env.cols):

            x = col * TILE_SIZE
            y = row * TILE_SIZE

            cell = env.grid[row, col]

            color = EMPTY_COLOR

            if cell == -1:
                color = WALL_COLOR

            elif cell == 2:
                color = TREASURE_COLOR

            elif cell == -2:
                color = TRAP_COLOR

            elif cell == 3:
                color = EXIT_COLOR

            pygame.draw.rect(
                screen,
                color,
                (x, y, TILE_SIZE, TILE_SIZE)
            )

            pygame.draw.rect(
                screen,
                GRID_COLOR,
                (x, y, TILE_SIZE, TILE_SIZE),
                2
            )

    # Draw hunter
    row, col = env.agent_pos

    center_x = col * TILE_SIZE + TILE_SIZE // 2
    center_y = row * TILE_SIZE + TILE_SIZE // 2

    pygame.draw.circle(
        screen,
        AGENT_COLOR,
        (center_x, center_y),
        28
    )

    hunter_text = title_font.render(
        "H",
        True,
        WHITE
    )

    text_rect = hunter_text.get_rect(
        center=(center_x, center_y)
    )

    screen.blit(hunter_text, text_rect)


# ==========================================
# 6. DRAW INFORMATION PANEL
# ==========================================

def draw_info():

    panel_y = TILE_SIZE * GRID_SIZE

    pygame.draw.rect(
        screen,
        WHITE,
        (0, panel_y, WINDOW_WIDTH, INFO_HEIGHT)
    )

    title = title_font.render(
        "TREASURE HUNTER - AI MODE",
        True,
        TEXT_COLOR
    )

    screen.blit(title, (12, panel_y + 8))

    info_text = (
        f"Steps: {env.steps}    "
        f"Reward: {last_reward:.1f}    "
        f"Total: {total_reward:.1f}"
    )

    info = info_font.render(
        info_text,
        True,
        TEXT_COLOR
    )

    screen.blit(info, (12, panel_y + 43))

    if env.treasure_collected:
        treasure_text = "Treasure: COLLECTED"
    else:
        treasure_text = "Treasure: NOT COLLECTED"

    treasure = small_font.render(
        treasure_text,
        True,
        TEXT_COLOR
    )

    screen.blit(treasure, (12, panel_y + 76))

    if paused:
        status_text = "PAUSED - Press SPACE to resume"

    elif not env.done:
        status_text = "AI PLAYING..."

    elif (
        env.agent_pos == env.exit_pos
        and env.treasure_collected
    ):
        status_text = "SUCCESS - Treasure collected!"

    elif env.agent_pos == env.trap_pos:
        status_text = "GAME OVER - AI hit the trap!"

    else:
        status_text = "EPISODE ENDED - Step limit reached"

    status = small_font.render(
        status_text,
        True,
        TEXT_COLOR
    )

    screen.blit(status, (12, panel_y + 103))

    controls = small_font.render(
        "R = Restart    SPACE = Pause    ESC = Quit",
        True,
        TEXT_COLOR
    )

    screen.blit(controls, (12, panel_y + 130))


# ==========================================
# 7. MAIN GAME LOOP
# ==========================================

while running:

    current_time = pygame.time.get_ticks()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_r:

                state = env.reset()

                last_reward = 0.0
                total_reward = 0.0

                paused = False

                last_move_time = pygame.time.get_ticks()

                print("Game restarted.")

            elif event.key == pygame.K_SPACE:

                if not env.done:
                    paused = not paused

    # --------------------------------------
    # AI CHOOSES AND EXECUTES AN ACTION
    # --------------------------------------

    if (
        running
        and not paused
        and not env.done
        and current_time - last_move_time >= MOVE_DELAY
    ):

        action = agent.select_action(
            state,
            training=False
        )

        next_state, reward, done, info = env.step(action)

        state = next_state

        last_reward = reward
        total_reward += reward

        last_move_time = current_time

        print(
            f"Step: {env.steps} | "
            f"Action: {action} | "
            f"Reward: {reward:.1f} | "
            f"Position: {env.agent_pos}"
        )

        if done:

            print("\nEpisode finished.")
            print("Treasure collected:", env.treasure_collected)
            print("Total reward:", round(total_reward, 2))

            if (
                env.treasure_collected
                and env.agent_pos == env.exit_pos
            ):
                print("RESULT: SUCCESS!")

            elif env.agent_pos == env.trap_pos:
                print("RESULT: TRAP HIT")

            else:
                print("RESULT: STEP LIMIT REACHED")

    # --------------------------------------
    # UPDATE DISPLAY
    # --------------------------------------

    draw_grid()
    draw_info()

    pygame.display.flip()

    clock.tick(60)


pygame.quit()
print("AI game closed.")
