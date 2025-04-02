To organize the Gobang game with alpha-beta pruning into a modular design following good engineering practices, we’ll split the code into separate Python files based on functionality. This improves readability, maintainability, and reusability. Below is the structure, with each file handling a distinct responsibility. We’ll use a 5x5 board and a win condition of three-in-a-row for simplicity, as in the previous example.

---

### Project Structure
```
gobang_game/
├── board.py         # Board initialization and display
├── game_rules.py    # Win checking and move generation
├── evaluation.py    # Heuristic evaluation of board states
├── ai.py            # Alpha-beta pruning and move selection
├── game.py          # Main game loop and player interaction
└── main.py          # Entry point to run the game
```

Each `.py` file will contain related functions, and we’ll use imports to connect them. Here’s how we’ll break it down:

---

### 1. `board.py` - Board Management
Handles the creation and display of the game board.

```python
# board.py
BOARD_SIZE = 5
EMPTY, BLACK, WHITE = 0, 1, 2

def create_board():
    """Initialize an empty 5x5 board."""
    return [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

def print_board(board):
    """Display the current state of the board."""
    for row in board:
        print(' '.join(str(cell) for cell in row))
    print()
```

- **Purpose**: Defines constants and manages the board’s state and visualization.
- **Why Separate**: Board-related logic is a distinct concern, reusable across different game implementations.

---

### 2. `game_rules.py` - Game Rules and Logic
Contains functions for checking wins and generating possible moves.

```python
# game_rules.py
from board import BOARD_SIZE, EMPTY, BLACK, WHITE

def check_win(board, player):
    """Check if the given player has three stones in a row."""
    # Check rows
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE - 2):
            if all(board[i][j + k] == player for k in range(3)):
                return True
    # Check columns
    for j in range(BOARD_SIZE):
        for i in range(BOARD_SIZE - 2):
            if all(board[i + k][j] == player for k in range(3)):
                return True
    # Check main diagonals
    for i in range(BOARD_SIZE - 2):
        for j in range(BOARD_SIZE - 2):
            if all(board[i + k][j + k] == player for k in range(3)):
                return True
    # Check anti-diagonals
    for i in range(BOARD_SIZE - 2):
        for j in range(2, BOARD_SIZE):
            if all(board[i + k][j - k] == player for k in range(3)):
                return True
    return False

def get_possible_moves(board):
    """Return a list of (row, col) tuples for all empty positions."""
    return [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if board[i][j] == EMPTY]
```

- **Purpose**: Encapsulates game-specific rules (winning conditions) and move generation.
- **Why Separate**: Rules are core to the game’s logic and may change (e.g., for a 15x15 board or five-in-a-row), so isolating them makes modifications easier.

---

### 3. `evaluation.py` - Board Evaluation
Defines the heuristic for scoring board states.

```python
# evaluation.py
from board import BOARD_SIZE, BLACK, WHITE

def evaluate(board):
    """Evaluate the board state based on two-in-a-row patterns."""
    black_score = count_patterns(board, BLACK, 2)
    white_score = count_patterns(board, WHITE, 2)
    return black_score - white_score

def count_patterns(board, player, length):
    """Count occurrences of 'length' stones in a row for the player."""
    count = 0
    # Check rows
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE - length + 1):
            if all(board[i][j + k] == player for k in range(length)):
                count += 1
    # Check columns
    for j in range(BOARD_SIZE):
        for i in range(BOARD_SIZE - length + 1):
            if all(board[i + k][j] == player for k in range(length)):
                count += 1
    # Check main diagonals
    for i in range(BOARD_SIZE - length + 1):
        for j in range(BOARD_SIZE - length + 1):
            if all(board[i + k][j + k] == player for k in range(length)):
                count += 1
    # Check anti-diagonals
    for i in range(BOARD_SIZE - length + 1):
        for j in range(length - 1, BOARD_SIZE):
            if all(board[i + k][j - k] == player for k in range(length)):
                count += 1
    return count
```

- **Purpose**: Provides the AI with a way to assess non-terminal board states.
- **Why Separate**: The evaluation function is a critical part of the AI’s decision-making and may need tuning or replacement (e.g., for a more complex heuristic).

---

### 4. `ai.py` - AI Logic with Alpha-Beta Pruning
Implements the minimax algorithm with alpha-beta pruning and move selection.

```python
# ai.py
from board import BLACK, WHITE, EMPTY
from game_rules import check_win, get_possible_moves
from evaluation import evaluate

def minimax(board, depth, is_maximizing, alpha, beta):
    """Minimax algorithm with alpha-beta pruning."""
    if check_win(board, BLACK):
        return 1000  # Black (AI) wins
    if check_win(board, WHITE):
        return -1000  # White (opponent) wins
    if depth == 0 or not get_possible_moves(board):
        return evaluate(board)

    if is_maximizing:  # AI's turn (Black)
        max_eval = float('-inf')
        for move in get_possible_moves(board):
            board[move[0]][move[1]] = BLACK
            eval = minimax(board, depth - 1, False, alpha, beta)
            board[move[0]][move[1]] = EMPTY
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:  # Opponent's turn (White)
        min_eval = float('inf')
        for move in get_possible_moves(board):
            board[move[0]][move[1]] = WHITE
            eval = minimax(board, depth - 1, True, alpha, beta)
            board[move[0]][move[1]] = EMPTY
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_best_move(board, depth):
    """Find the best move for the AI (Black)."""
    best_eval = float('-inf')
    best_move = None
    alpha, beta = float('-inf'), float('inf')
    for move in get_possible_moves(board):
        board[move[0]][move[1]] = BLACK
        eval = minimax(board, depth - 1, False, alpha, beta)
        board[move[0]][move[1]] = EMPTY
        if eval > best_eval:
            best_eval = eval
            best_move = move
        alpha = max(alpha, eval)
    return best_move
```

- **Purpose**: Contains all AI-specific logic, including the core alpha-beta pruning algorithm.
- **Why Separate**: AI behavior is a distinct module that could be swapped (e.g., for a different algorithm) without affecting other components.

---

### 5. `game.py` - Game Loop and Interaction
Manages the game flow and player interaction.

```python
# game.py
from board import create_board, print_board, BLACK, WHITE
from game_rules import check_win, get_possible_moves
from ai import get_best_move

def play_game():
    """Run the game with AI vs. human."""
    board = create_board()
    depth = 3  # Search depth for AI
    current_player = BLACK  # AI starts

    while True:
        print_board(board)
        if current_player == BLACK:
            print("AI's turn...")
            move = get_best_move(board, depth)
            if move:
                board[move[0]][move[1]] = BLACK
                print(f"AI placed at {move}")
                if check_win(board, BLACK):
                    print_board(board)
                    print("AI wins!")
                    break
        else:
            print("Your turn...")
            try:
                x, y = map(int, input("Enter move (row col): ").split())
                if (0 <= x < len(board) and 0 <= y < len(board[0]) and 
                    board[x][y] == 0):
                    board[x][y] = WHITE
                    if check_win(board, WHITE):
                        print_board(board)
                        print("You win!")
                        break
                else:
                    print("Invalid move!")
                    continue
            except (ValueError, IndexError):
                print("Invalid input! Enter two numbers (e.g., '2 3').")
                continue

        current_player = WHITE if current_player == BLACK else BLACK
        if not get_possible_moves(board):
            print("Game over: Draw!")
            break
```

- **Purpose**: Orchestrates the game, handling turns and user input.
- **Why Separate**: The game loop ties everything together and could be adapted for different interfaces (e.g., GUI) or game modes.

---

### 6. `main.py` - Entry Point
The starting point to launch the game.

```python
# main.py
from game import play_game

if __name__ == "__main__":
    play_game()
```

- **Purpose**: Provides a clean entry point, keeping the project runnable with a single command (`python main.py`).
- **Why Separate**: Standard practice to separate execution from logic, allowing `game.py` to be imported elsewhere if needed.

---

### How to Run
1. Create a directory called `gobang_game`.
2. Save each code block in its respective `.py` file inside that directory.
3. Navigate to the directory in your terminal.
4. Run `python main.py` to start the game.

---

### Example Interaction
```
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0

AI's turn...
AI placed at (2, 2)
0 0 0 0 0
0 0 0 0 0
0 0 1 0 0
0 0 0 0 0
0 0 0 0 0

Your turn...
Enter move (row col): 2 1
0 0 0 0 0
0 0 0 0 0
0 2 1 0 0
0 0 0 0 0
0 0 0 0 0

AI's turn...
```

---

### Modular Design Benefits
- **Separation of Concerns**: Each module handles one aspect (board, rules, evaluation, AI, game flow), making it easier to debug or extend.
- **Reusability**: Functions like `check_win` or `evaluate` could be reused in a different Gobang variant or game.
- **Maintainability**: Changes (e.g., tweaking the AI depth or adding a GUI) are confined to specific files.
- **Scalability**: Adding features (e.g., transposition tables, GUI) involves modifying or adding modules without rewriting everything.

This structure aligns with engineering habits, providing a clean, professional layout for a small game project. You can now expand it—say, to a 15x15 board or a more sophisticated evaluation function—by editing the relevant modules!