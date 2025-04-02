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