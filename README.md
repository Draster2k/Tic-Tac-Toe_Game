# Tic-Tac-Toe Game Using Pygame

## Description
This project is a simple implementation of the classic Tic-Tac-Toe game using Pygame. It allows two players to play the game, displaying the current player's move and checking for wins or draws. When a player wins or the game ends in a draw, a message is displayed on the screen.

## Setup and Installation
1. Make sure you have Python installed.
2. Install Pygame using:
   ```
   pip install pygame
   ```
3. Run the game using:
   ```
   python tic_tac_toe.py
   ```

## Game Mechanics
The game consists of a 3x3 grid where players take turns placing their symbol (X or O) in empty cells. The game checks for a win when three of the same symbols are aligned horizontally, vertically, or diagonally, or it checks for a draw if all cells are filled.

## The Game Code:
```python
import pygame
import sys

#start pygame
pygame.init()

#screen settings
screen_width=600
screen_hight=600
line_color=(0,0,0)
bg_color=(255,255,255)
line_width=15

#create the screen
screen=pygame.display.set_mode((screen_width,screen_hight))
pygame.display.set_caption("Tic-Tac-Toe")

#set the background color
screen.fill(bg_color)

#Game Variables
board=[[None for _ in range(3)] for _ in range (3)] #3x3 grid
current_player="X" #Starts with player X

#draw the grid lines
def draw_grid():
    #Vertical Lines
    pygame.draw.line(screen,line_color,(200,0),(200,screen_hight),line_width)
    pygame.draw.line(screen,line_color,(400,0),(400,screen_width),line_width)
    #Horisontal Lines
    pygame.draw.line(screen,line_color,(0,200),(screen_hight,200),line_width)
    pygame.draw.line(screen,line_color,(0,400),(screen_hight,400),line_width)

def handle_click(x,y):
    #determens row and column based on the click position
    row=y//200
    col=x//200

    #place the current player's symbol if the cell is empty
    if board[row][col] is None:
        board[row][col]=current_player
        return True
    return False

def switch_player():
    global current_player
    current_player="O" if current_player=="X" else "X"

def draw_symbols():
    for row in range(3):
        for col in range(3):
            if board[row][col] == "X":
                # Draw X
                pygame.draw.line(screen, (255, 0, 0), (col * 200 + 50, row * 200 + 50), (col * 200 + 150, row * 200 + 150), line_width)
                pygame.draw.line(screen, (255, 0, 0), (col * 200 + 150, row * 200 + 50), (col * 200 + 50, row * 200 + 150), line_width)
            elif board[row][col] == "O":
                # Draw O
                pygame.draw.circle(screen, (0, 0, 255), (col * 200 + 100, row * 200 + 100), 70, line_width)

def check_win():
    # Check rows for a win
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] and board[row][0] is not None:
            return True

    # Check columns for a win
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] is not None:
            return True

    # Check diagonals for a win
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None:
        return True
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None:
        return True

    return False

def check_draw():
    for row in board:
        if None in row:
            return False  # There's at least one empty cell, so not a draw
    return True  # No empty cells found, it's a draw

def show_message(message):
    # Create a text surface
    text_surface = font.render(message, True, (0, 0, 0))  # Black text
    text_rect = text_surface.get_rect(center=(screen_width // 2, screen_hight // 2))

    # Draw the text on the screen
    screen.blit(text_surface, text_rect)
    pygame.display.update()

# Initialize font
pygame.font.init()
font = pygame.font.Font(None, 80)  # You can change the font size as needed

#main game loop
def main():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouseX = event.pos[0]  # x-coordinate of the click
                mouseY = event.pos[1]  # y-coordinate of the click

                # Make a move if possible
                if handle_click(mouseX, mouseY):
                    draw_symbols()

                    # Check for a win
                    if check_win():
                        show_message(f"Player {current_player} wins!")
                        pygame.time.wait(2000)
                        reset_game()
                    # Check for a draw
                    elif check_draw():
                        show_message("It's a draw!")
                        pygame.time.wait(2000)
                        reset_game()

                    switch_player()

        screen.fill(bg_color)
        draw_grid()
        draw_symbols()
        pygame.display.update()

def reset_game():
    global board, current_player
    board = [[None for _ in range(3)] for _ in range(3)]
    current_player = "X"

if __name__=="__main__":
    main()
```
## Steps to Create the Tic-Tac-Toe Game

1. **Set Up the Project Environment**
   - Install Python if not already installed.
   - Install Pygame using `pip install pygame`.
   - Create a new Python file, e.g., `tic_tac_toe.py`.

2. **Initialize Pygame and Set Up the Window**
   - Import the necessary libraries (`pygame` and `sys`).
   - Initialize Pygame with `pygame.init()`.
   - Set up the screen dimensions and create a Pygame window using `pygame.display.set_mode()`.

3. **Define Game Variables**
   - Set up variables for screen dimensions, colors, and line widths.
   - Initialize a 3x3 grid to represent the game board.
   - Create a variable to track the current player (e.g., "X" or "O").

4. **Draw the Grid**
   - Write a function `draw_grid()` to draw vertical and horizontal lines, dividing the screen into a 3x3 grid.

5. **Handle Player Input**
   - Add event handling for mouse clicks.
   - Write a function `handle_click(x, y)` to determine which cell was clicked and update the board if the cell is empty.

6. **Draw X and O Symbols on the Board**
   - Create a `draw_symbols()` function to iterate over the board and draw the appropriate symbol (X or O) based on the board’s state.

7. **Switch Between Players**
   - Add functionality to switch the current player after each valid move.

8. **Check for a Win Condition**
   - Implement a `check_win()` function that checks if a player has three of the same symbols in a row, column, or diagonal.

9. **Check for a Draw Condition**
   - Implement a `check_draw()` function to detect if all cells are filled and no player has won.

10. **Display the Result**
    - Set up a font for rendering text messages.
    - Write a `show_message(message)` function to display a message indicating the winner or a draw.

11. **Reset the Game**
    - Implement a `reset_game()` function to reset the board and switch back to the starting player after a win or draw.

12. **Create the Main Game Loop**
    - In the main loop, handle events for quitting, mouse clicks, and updating the screen.
    - Check for win or draw conditions after each move.
    - Call functions to draw the grid, symbols, and display the result if the game ends.

13. **Test the Game**
    - Run the game to ensure all features work as expected.
    - Debug any issues and make necessary adjustments.

14. **Document the Project**
    - Create a `README.md` file to document the game, its features, and how to run it.
    - Include explanations for key sections of the code and list any future improvements.
