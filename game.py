import pygame
from environment import TreasureHunterEnv


# -----------------------------
# PYGAME SETTINGS
# -----------------------------

pygame.init()

TILE_SIZE = 90
GRID_SIZE = 6
INFO_HEIGHT = 105

WINDOW_WIDTH = TILE_SIZE * GRID_SIZE
WINDOW_HEIGHT = TILE_SIZE * GRID_SIZE + INFO_HEIGHT

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Treasure Hunter - RL Environment")

clock = pygame.time.Clock()

# Fonts
title_font = pygame.font.Font(None, 36)
info_font = pygame.font.Font(None, 28)
small_font = pygame.font.Font(None, 24)


# -----------------------------
# COLORS
# -----------------------------

BACKGROUND = (235, 235, 235)
GRID_COLOR = (180, 180, 180)

EMPTY_COLOR = (245, 245, 245)
WALL_COLOR = (45, 45, 45)
TREASURE_COLOR = (245, 190, 40)
TRAP_COLOR = (220, 70, 70)
EXIT_COLOR = (70, 180, 100)
AGENT_COLOR = (70, 120, 220)

TEXT_COLOR = (30, 30, 30)
WHITE = (255, 255, 255)


# -----------------------------
# ENVIRONMENT
# -----------------------------

env = TreasureHunterEnv()


# -----------------------------
# DRAW GRID
# -----------------------------

def draw_grid():
    """Draw the current game grid."""

    screen.fill(BACKGROUND)

    for row in range(env.rows):
        for col in range(env.cols):

            x = col * TILE_SIZE
            y = row * TILE_SIZE

            cell = env.grid[row, col]

            # Default empty cell
            color = EMPTY_COLOR

            # Wall
            if cell == -1:
                color = WALL_COLOR

            # Treasure
            elif cell == 2:
                color = TREASURE_COLOR

            # Trap
            elif cell == -2:
                color = TRAP_COLOR

            # Exit
            elif cell == 3:
                color = EXIT_COLOR

            pygame.draw.rect(
                screen,
                color,
                (x, y, TILE_SIZE, TILE_SIZE)
            )

            # Grid border
            pygame.draw.rect(
                screen,
                GRID_COLOR,
                (x, y, TILE_SIZE, TILE_SIZE),
                2
            )

    # -------------------------
    # DRAW AGENT
    # -------------------------

    agent_row, agent_col = env.agent_pos

    center_x = agent_col * TILE_SIZE + TILE_SIZE // 2
    center_y = agent_row * TILE_SIZE + TILE_SIZE // 2

    pygame.draw.circle(
        screen,
        AGENT_COLOR,
        (center_x, center_y),
        28
    )

    # Draw "H" for Hunter
    hunter_text = title_font.render("H", True, WHITE)

    text_rect = hunter_text.get_rect(
        center=(center_x, center_y)
    )

    screen.blit(hunter_text, text_rect)


# -----------------------------
# DRAW INFORMATION PANEL
# -----------------------------

def draw_info(last_reward):
    """Draw the information below the grid."""

    panel_y = TILE_SIZE * GRID_SIZE

    pygame.draw.rect(
        screen,
        WHITE,
        (0, panel_y, WINDOW_WIDTH, INFO_HEIGHT)
    )

    # Title
    title = title_font.render(
        "TREASURE HUNTER",
        True,
        TEXT_COLOR
    )

    screen.blit(title, (15, panel_y + 10))

    # Game information
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

    screen.blit(info, (15, panel_y + 48))

    # Treasure status
    if env.treasure_collected:
        treasure_text = "Treasure: COLLECTED"
    else:
        treasure_text = "Treasure: NOT COLLECTED"

        # Treasure status
    if env.treasure_collected:
        treasure_text = "Treasure: COLLECTED"
    else:
        treasure_text = "Treasure: NOT COLLECTED"

    treasure = small_font.render(
        treasure_text,
        True,
        TEXT_COLOR
    )

    screen.blit (
        treasure,
        (15, panel_y + 80)
    )

    # Status
    if env.done:

        if env.agent_pos == env.trap_pos:
            status_text = "GAME OVER - You hit the trap!"

        elif (
            env.agent_pos == env.exit_pos
            and env.treasure_collected
        ):
            status_text = "SUCCESS - Treasure collected and exit reached!"

        else:
            status_text = "EPISODE ENDED - Maximum steps reached"

    else:
        status_text = "Use ARROW KEYS | R = Restart | ESC = Quit"

    status = small_font.render(
        status_text,
        True,
        TEXT_COLOR
    )

    screen.blit (
        status,
        (15, panel_y + 105)
    )


# -----------------------------
# MAIN GAME LOOP
# -----------------------------

running = True
last_reward = 0.0
total_reward = 0.0

while running:

    # -------------------------
    # EVENTS
    # -------------------------

    for event in pygame.event.get():

        # Close window
        if event.type == pygame.QUIT:
            running = False

        # Keyboard
        if event.type == pygame.KEYDOWN:

            # Quit
            if event.key == pygame.K_ESCAPE:
                running = False

            # Restart
            elif event.key == pygame.K_r:

                env.reset()

                last_reward = 0.0
                total_reward = 0.0

            # Movement
            elif not env.done:

                action = None

                if event.key == pygame.K_UP:
                    action = 0

                elif event.key == pygame.K_DOWN:
                    action = 1

                elif event.key == pygame.K_LEFT:
                    action = 2

                elif event.key == pygame.K_RIGHT:
                    action = 3

                # Perform action
                if action is not None:

                    _, reward, done, info = env.step(action)

                    last_reward = reward
                    total_reward += reward

    # -------------------------
    # DRAW EVERYTHING
    # -------------------------

    draw_grid()
    draw_info(last_reward)

    pygame.display.flip()

    # Limit FPS
    clock.tick(60)


pygame.quit()