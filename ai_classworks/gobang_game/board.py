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