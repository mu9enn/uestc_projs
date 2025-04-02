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