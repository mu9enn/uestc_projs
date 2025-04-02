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