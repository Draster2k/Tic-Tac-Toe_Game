# Imports
import os
import matplotlib.pyplot as plt
import numpy as np
import pygame
import random
import torch
import logging
import multiprocessing
import seaborn as sns
from torchviz import make_dot
from DQL import DQLAgent

# Pygame Initialization
pygame.init()

os.makedirs("Logs", exist_ok=True)
os.makedirs("NN", exist_ok=True)

# Global Screen Settings
screen_width, screen_height = 1600, 900  # Combined view dimensions
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("AI Training")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
line_color = BLACK
bg_color = WHITE

# Font settings
font = pygame.font.Font(None, 40)

# Global game variables
board = [[None] * 3 for _ in range(3)]
current_player = "X"

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("Logs/training_log.log"),
        logging.StreamHandler()
    ]
)

# Constants for combined view positioning
NN_OFFSET_X = 0
NN_OFFSET_Y = 0
GAME_OFFSET_X = 800
GAME_OFFSET_Y = 0

# Initialize rewards list
rewards = []

# Plot rewards after training
def plot_rewards(rewards):
    if not isinstance(rewards, list):
        logging.error(f"plot_rewards expected a list, but got {type(rewards).__name__}")
        return
    cumulative_rewards = np.cumsum(rewards)
    print(f"Plotting {len(cumulative_rewards)} cumulative rewards.")
    plt.figure(figsize=(16, 8))
    plt.plot(cumulative_rewards, label='Cumulative Reward', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Training Rewards over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    plt.pause(0.001)

# Draw text on the screen
def draw_text(text, x, y):
    text_surface = font.render(text, True, BLACK)
    screen.blit(text_surface, (x, y))

# Train single AI
def train_single_ai():
    process = multiprocessing.Process(target=train_ai_with_nn)
    process.start()  # Run the process asynchronously

# Draw the grid
def draw_grid(offset_x=500, offset_y=150):
    for i in range(1, 3):  # Draw vertical lines
        pygame.draw.line(screen, BLACK, (offset_x + i * 200, offset_y),
                         (offset_x + i * 200, offset_y + 600), 15)
    for i in range(1, 3):  # Draw horizontal lines
        pygame.draw.line(screen, BLACK, (offset_x, offset_y + i * 200),
                         (offset_x + 600, offset_y + i * 200), 15)


def draw_symbols(offset_x=500, offset_y=150):
    for row in range(3):
        for col in range(3):
            cell_x = offset_x + col * 200
            cell_y = offset_y + row * 200
            if board[row][col] == "X":
                pygame.draw.line(screen, (255, 0, 0),
                                 (cell_x + 50, cell_y + 50),
                                 (cell_x + 150, cell_y + 150), 15)
                pygame.draw.line(screen, (255, 0, 0),
                                 (cell_x + 150, cell_y + 50),
                                 (cell_x + 50, cell_y + 150), 15)
            elif board[row][col] == "O":
                pygame.draw.circle(screen, (0, 0, 255),
                                   (cell_x + 100, cell_y + 100), 70, 15)

# Check for a win
def check_win():
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] and board[row][0] is not None:
            return True
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] is not None:
            return True
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None:
        return True
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None:
        return True
    return False

# Check for a draw
def check_draw():
    for row in board:
        if None in row:
            return False
    return True

# Check for a win on a custom board
def check_win_custom(board):
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] and board[row][0] is not None:
            return True
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] is not None:
            return True
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None:
        return True
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None:
        return True
    return False

# Check for a draw on a custom board
def check_draw_custom(board):
    for row in board:
        if None in row:
            return False
    return True

# Draw the game container
def draw_grid_in_container(x_offset, y_offset, board):
    for row in range(3):
        for col in range(3):
            # Adjust cell size to 200x200
            cell_x = x_offset + col * 200
            cell_y = y_offset + row * 200

            # Draw grid cell border
            pygame.draw.rect(screen, BLACK, (cell_x, cell_y, 200, 200), 5)

            # Draw X and O symbols
            if board[row][col] == "X":
                pygame.draw.line(screen, (255, 0, 0), (cell_x + 20, cell_y + 20), (cell_x + 180, cell_y + 180), 10)
                pygame.draw.line(screen, (255, 0, 0), (cell_x + 180, cell_y + 20), (cell_x + 20, cell_y + 180), 10)
            elif board[row][col] == "O":
                pygame.draw.circle(screen, (0, 0, 255), (cell_x + 100, cell_y + 100), 80, 10)

# Visualize weights of the neural network layers
def visualize_weights(agent):
    try:
        if hasattr(agent, 'model'):
            for name, param in agent.model.named_parameters():
                if param.requires_grad:
                    plt.figure(figsize=(8, 6))
                    plt.title(f"Weights of layer: {name}")
                    plt.hist(param.data.numpy().flatten(), bins=30, color='skyblue', edgecolor='black')
                    plt.xlabel('Weight Values')
                    plt.ylabel('Frequency')
                    plt.grid(True)
                    plt.show()
    except Exception as e:
        logging.error(f"Failed to visualize weights: {e}")

# Visualize the neural network architecture
def visualize_nn_architecture(agent, episode):
    # Create a dummy input representing the board state
    dummy_input = torch.zeros((1, 9))

    # Generate the computational graph
    model_graph = make_dot(agent.model(dummy_input), params=dict(agent.model.named_parameters()))

    # Set the graph attributes for size
    model_graph.graph_attr.update(size='8,6')  # Adjust graph size

    # Define the output path with a unique name for each episode
    output_file = f"nn_architecture_episode_{episode}.png"

    # Ensure no directory with the same name exists
    if os.path.isdir(output_file):
        os.rmdir(output_file)

    # Render the graph
    model_graph.render(output_file.replace(".png", ""), format="png", cleanup=True)
    model_graph.view()

# 2 Player mode
def two_player_mode():
    global current_player
    reset_game()
    running = True

    while running:
        screen.fill(bg_color)
        draw_grid()
        draw_symbols()
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()

                if 500 <= x <= 1100 and 150 <= y <= 750:
                    row = (y - 150) // 200
                    col = (x - 500) // 200

                    if board[row][col] is None:
                        board[row][col] = current_player

                        if check_win():
                            show_message(f"{current_player} wins!")
                            pygame.time.wait(2000)
                            reset_game()
                            return
                        elif check_draw():
                            show_message("It's a draw!")
                            pygame.time.wait(2000)
                            reset_game()
                            return

                        current_player = "O" if current_player == "X" else "X"

# VS AI mode
def vs_ai_mode():
    global current_player
    agent = DQLAgent()
    reset_game()
    running = True

    while running:
        screen.fill(bg_color)
        draw_grid()
        draw_symbols()
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()

            if event.type == pygame.MOUSEBUTTONDOWN and current_player == "X":
                x, y = pygame.mouse.get_pos()

                if 500 <= x <= 1100 and 150 <= y <= 750:
                    row = (y - 150) // 200
                    col = (x - 500) // 200

                    if board[row][col] is None:
                        board[row][col] = current_player

                        if check_win():
                            show_message(f"{current_player} wins!")
                            pygame.time.wait(2000)
                            reset_game()
                            return
                        elif check_draw():
                            show_message("It's a draw!")
                            pygame.time.wait(2000)
                            reset_game()
                            return

                    current_player = "O"

        if current_player == "O":
            state = agent.get_board_state(board)
            action = agent.choose_action(state)
            row, col = action // 3, action % 3

            if board[row][col] is None:
                board[row][col] = current_player

                if check_win():
                    show_message("O wins!")
                    pygame.time.wait(2000)
                    reset_game()
                    return
                elif check_draw():
                    show_message("It's a draw!")
                    pygame.time.wait(2000)
                    reset_game()
                    return

                current_player = "X"

# Main menu function
def main_menu():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return "exit"

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return "2player"
                elif event.key == pygame.K_2:
                    return "vs_ai"
                elif event.key == pygame.K_3:
                    return "train_1_ai"
                elif event.key == pygame.K_4:
                    return "train_8_ai"
                elif event.key == pygame.K_5:
                    return "Show NN Weights"
                elif event.key == pygame.K_6:
                    return "Show NN Graph"
                elif event.key == pygame.K_7:
                    pygame.quit()
                    return "exit"

        screen.fill(bg_color)
        center_x = 800  # Centered horizontally at 1600 / 2

        draw_text("1. 2 Player", center_x - 100, 200)
        draw_text("2. Vs AI", center_x - 100, 250)
        draw_text("3. Train 1 AI", center_x - 100, 300)
        draw_text("4. Train 8 AIs", center_x - 100, 350)
        draw_text("5. Show NN Weights", center_x - 100, 400)
        draw_text("6. Show NN Graph", center_x - 100, 450)
        draw_text("7. Exit", center_x - 100, 500)

        pygame.display.update()

# Reset the game
def reset_game():
    global board, current_player
    board = [[None] * 3 for _ in range(3)]
    current_player = "X"

# Show message on the screen
def show_message(message):
    screen.fill(WHITE)
    draw_text(message, 700, 400)  # Centered around (800, 450)
    pygame.display.update()

def play_game(agent, opponent):
    board = reset_game()
    current_player = "X"
    done = False

    while not done:
        if current_player == "X":
            action = agent.choose_action(agent.get_board_state(board))
        else:
            action = opponent.choose_action(agent.get_board_state((board)))
        board, reward, done = agent.choose_action(board, action)

        current_player = "O" if current_player =="X" else "X"
        if done:
            return "win" if reward == 1 else "loss" if reward == -1 else "draw"
'''
# Combined game and NN visualization during training (Working)
def train_ai_with_nn():
    global board, current_player
    screen_width, screen_height = 1600, 900  # Larger window to match the main screen
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("AI Training")

    agent = DQLAgent()
    reset_game()
    max_moves = 20
    cumulative_rewards = []
    total_episodes = 100
    current_player = "X"

    for episode in range(total_episodes):
        reset_game()
        done = False
        move_count = 0
        episode_reward = 0

        while not done and move_count < max_moves:
            state = agent.get_board_state(board)

            # Update game display on the right
            screen.fill(bg_color)
            draw_grid(offset_x=800)  # Centered around 1600px width
            draw_symbols(offset_x=800)
            draw_text(f"Training Episode: {episode + 1}/{total_episodes}", 620, 10)

            # Render NN visualization on the left
            draw_nn_visualization(agent)

            pygame.display.update()

            # AI's turn ('O')
            if current_player == "O":
                valid_actions = [i for i in range(9) if board[i // 3][i % 3] is None]

                if not valid_actions:
                    logging.info(f"Episode {episode + 1}: No valid actions left, ending episode.")
                    done = True
                else:
                    action = agent.choose_action(state, valid_actions)
                    row, col = action // 3, action % 3

                    if board[row][col] is None:
                        board[row][col] = current_player

                        if check_win():
                            reward = 1
                            done = True
                            logging.info(f"Episode {episode + 1}: AI (O) won.")
                        elif check_draw():
                            reward = 0
                            done = True
                            logging.info(f"Episode {episode + 1}: Game ended in a draw.")
                        else:
                            reward = 0
                            current_player = "X"

                        next_state = agent.get_board_state(board)
                        agent.store_experience(state, action, reward, next_state, done)
                        agent.replay()
                        episode_reward += reward

            # Random move for 'X'
            elif current_player == "X":
                action = random.choice([i for i in range(9) if board[i // 3][i % 3] is None])
                row, col = action // 3, action % 3

                if board[row][col] is None:
                    board[row][col] = current_player

                    if check_win():
                        reward = -1  # Negative reward for AI
                        done = True
                        logging.info(f"Episode {episode + 1}: Player (X) won.")
                    elif check_draw():
                        reward = 0
                        done = True
                        logging.info(f"Episode {episode + 1}: Game ended in a draw.")
                    else:
                        reward = 0
                        current_player = "O"

            move_count += 1

        # Log cumulative reward for this episode
        cumulative_rewards.append(episode_reward)

        if move_count >= max_moves:
            logging.warning(f"Episode {episode + 1}: Reached max moves, forced termination.")

    # Plot cumulative rewards once after training
    plt.plot(cumulative_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Training Progress")
    plt.show()

    # Evaluation after training
    random_agent = RandomAgent(agent)  # Instantiate random agent with the DQL agent
    wins, losses, draws = 0, 0, 0
    test_games = 100

    for game in range(test_games):
        result = play_game(agent, random_agent)
        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        else:
            draws += 1

    logging.info("AI Training completed.")
    logging.info(f"Evaluation results: Wins: {wins}, Losses: {losses}, Draws: {draws}")
'''

def train_ai_with_nn():
    global board, current_player
    screen_width, screen_height = 1600, 900
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("AI Training")

    agent = DQLAgent()
    reset_game()
    max_moves = 40
    cumulative_rewards = []
    total_episodes = 100
    current_player = "X"

    # Initialize W/L/D counters
    win_count, loss_count, draw_count = 0, 0, 0

    for episode in range(total_episodes):
        reset_game()
        done = False
        move_count = 0
        episode_reward = 0

        while not done and move_count < max_moves:
            state = agent.get_board_state(board)

            # Fill the background and draw main elements
            screen.fill(bg_color)
            draw_grid(offset_x=800)
            draw_symbols(offset_x=800)
            draw_text(f"Training Episode: {episode + 1}/{total_episodes}", 620, 10)

            # Display W/L/D counters
            draw_text(f"Wins: {win_count}", 620, 50)
            draw_text(f"Losses: {loss_count}", 620, 90)
            draw_text(f"Draws: {draw_count}", 620, 130)


            # Render neural network visualization
            draw_nn_visualization(agent)

            # Update the display for each frame
            pygame.display.flip()  # Use `flip()` for a complete screen update

            # AI's turn logic and counter updates
            if current_player == "O":
                valid_actions = [i for i in range(9) if board[i // 3][i % 3] is None]

                if not valid_actions:
                    done = True
                else:
                    action = agent.choose_action(state, valid_actions)
                    row, col = action // 3, action % 3

                    if board[row][col] is None:
                        board[row][col] = current_player

                        if check_win():
                            reward = 1
                            win_count += 1  # Increment win counter
                            done = True
                        elif check_draw():
                            reward = 0
                            draw_count += 1  # Increment draw counter
                            done = True
                        else:
                            reward = 0
                            current_player = "X"

                        next_state = agent.get_board_state(board)
                        agent.store_experience(state, action, reward, next_state, done)
                        agent.replay()
                        episode_reward += reward

            elif current_player == "X":
                action = random.choice([i for i in range(9) if board[i // 3][i % 3] is None])
                row, col = action // 3, action % 3

                if board[row][col] is None:
                    board[row][col] = current_player

                    if check_win():
                        reward = -1
                        loss_count += 1  # Increment loss counter
                        done = True
                    elif check_draw():
                        reward = 0
                        draw_count += 1  # Increment draw counter
                        done = True
                    else:
                        reward = 0
                        current_player = "O"

            move_count += 1

        cumulative_rewards.append(episode_reward)

    # Plot cumulative rewards after training
    plt.plot(cumulative_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Training Progress")
    plt.show()

    logging.info(f"Training completed. Results - Wins: {win_count}, Losses: {loss_count}, Draws: {draw_count}")

# Draw neural network visualization
def draw_nn_visualization(agent):
    try:
        # Create dummy input for the NN visualization
        dummy_input = torch.zeros((1, 9))

        # Generate the computational graph
        model_graph = make_dot(agent.model(dummy_input), params=dict(agent.model.named_parameters()))

        model_graph.graph_attr.update(size='8,6')

        # Save and display the graph on the left side
        graph_filename = 'NN/nn_architecture'
        model_graph.render(graph_filename, format='png', cleanup=True)
        img = pygame.image.load(f"{graph_filename}.png")
        img = pygame.transform.scale(img, (600, 600))
        screen.blit(img, (0, 150))
    except Exception as e:
        logging.error(f"Failed to render NN visualization: {e}")

# Train multiple AIs in a single window
def train_multiple_ais_in_single_window(num_instances=8):
    global screen, screen_width, screen_height, rewards

    screen_width, screen_height = 1600, 900
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Training 8 AIs in Parallel")
    rewards = []

    agents = [DQLAgent() for _ in range(num_instances)]  # Initialize 8 AIs
    boards = [[[None] * 3 for _ in range(3)] for _ in range(num_instances)]  # Create 8 boards
    current_players = ["O"] * num_instances  # Start with 'O' for all

    total_episodes = 1000
    max_moves = 20

    grid_width, grid_height = 400, 400  # Adjusted cell size for 4x2 grid layout
    cols = 4
    rows = 2

    # Calculate the vertical offset to center the grid
    y_offset_base = (screen_height - (rows * grid_height)) // 2

    episode = 0
    running = True

    while running and episode < total_episodes:
        boards = [[[None] * 3 for _ in range(3)] for _ in range(num_instances)]
        done = [False] * num_instances
        move_count = 0
        episode_result = ["draw"] * num_instances  # Track result for each AI

        # Capture initial Q-values for each agent before the episode
        old_q_values_list = [
            {name: param.data.clone() for name, param in agent.model.named_parameters() if
             'weight' in name or 'bias' in name}
            for agent in agents
        ]

        while not all(done) and move_count < max_moves:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return  # Exit training loop gracefully
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        running = False  # Stop training when space bar is pressed

            screen.fill((255, 255, 255))

            for i, agent in enumerate(agents):
                if done[i]:
                    continue

                # Calculate x and y offsets based on the grid position
                x_offset = (i % cols) * grid_width
                y_offset = y_offset_base + (i // cols) * grid_height

                # Draw each AI's grid in its designated area
                draw_grid_in_container(x_offset, y_offset, boards[i])

                board = boards[i]
                state = agent.get_board_state(board)

                if current_players[i] == "O":
                    valid_actions = [j for j in range(9) if board[j // 3][j % 3] is None]

                    if valid_actions:
                        action = agent.choose_action(state, valid_actions)
                        row, col = action // 3, action % 3

                        if board[row][col] is None:
                            board[row][col] = "O"

                            if check_win_custom(board):
                                reward = 1
                                done[i] = True
                                episode_result[i] = "win"
                            elif check_draw_custom(board):
                                reward = 0
                                done[i] = True
                                episode_result[i] = "draw"
                            else:
                                reward = 0
                                current_players[i] = "X"

                            rewards.append(reward)
                            next_state = agent.get_board_state(board)
                            agent.store_experience(state, action, reward, next_state, done[i])
                            agent.replay()

                elif current_players[i] == "X":
                    valid_actions = [j for j in range(9) if board[j // 3][j % 3] is None]

                    if valid_actions:
                        action = random.choice(valid_actions)
                        row, col = action // 3, action % 3

                        if board[row][col] is None:
                            board[row][col] = "X"

                            if check_win_custom(board):
                                reward = -1
                                done[i] = True
                                episode_result[i] = "loss"
                            elif check_draw_custom(board):
                                reward = 0
                                done[i] = True
                                episode_result[i] = "draw"
                            else:
                                reward = 0
                                current_players[i] = "O"

                            rewards.append(reward)

            move_count += 1
            pygame.display.flip()

        # Compare Q-values after the episode and log the differences
        for i, agent in enumerate(agents):
            q_differences = agent.compare_q_tables(old_q_values_list[i])
            logging.info(f"Episode {episode + 1}, Agent {i + 1}: Q-value differences: {q_differences}")

        episode += 1

    plot_heatmap_and_rewards(rewards, agents)
    logging.info("Training completed for all AIs.")

# Plotting function for heatmap and rewards
def plot_heatmap_and_rewards(rewards, agents):
    # Collect action counts for heatmap
    action_counts = [0] * 9
    for agent in agents:
        for _ in range(1000):
            state = agent.get_board_state([[None] * 3 for _ in range(3)])
            valid_actions = [i for i in range(9)]
            action = agent.choose_action(state, valid_actions)
            action_counts[action] += 1

    # Prepare cumulative rewards
    cumulative_rewards = [sum(rewards[:i+1]) for i in range(len(rewards))]

    # Create a single figure with two subplots
    plt.figure(figsize=(12, 6))

    # Subplot 1: Action Selection Heatmap
    plt.subplot(1, 2, 1)
    heatmap_data = [action_counts[:3], action_counts[3:6], action_counts[6:]]
    sns.heatmap(heatmap_data, annot=True, cmap='rainbow', fmt='d')
    plt.title('Action Selection Heatmap')
    plt.xlabel('Column')
    plt.ylabel('Row')

    # Subplot 2: Cumulative Rewards Over Time
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_rewards, label='Cumulative Reward')
    plt.title('Cumulative Rewards Over Time')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid()

    # Show the combined plot
    plt.tight_layout()
    plt.show()

    logging.info("Training completed for all AIs.")

# Main game loop
def main():
    while True:
        mode = main_menu()

        if mode == "2player":
            two_player_mode()
        elif mode == "vs_ai":
            vs_ai_mode()
        elif mode == "train_1_ai":
            print("Training 1 AI in a separate process...")
            train_single_ai()
        elif mode == "train_8_ai":
            print("Training 8 AIs in a single large window...")
            train_multiple_ais_in_single_window(8)
        elif mode == "Show NN Weights":
            visualize_weights(DQLAgent())  # Pass a DQLAgent instance
        elif mode == "Show NN Graph":
            visualize_nn_architecture(DQLAgent(), episode=1)  # Pass a DQLAgent instance
        elif mode == "exit":
            break

    pygame.quit()  # Ensure Pygame quits cleanly when exiting

if __name__ == "__main__":
    main()