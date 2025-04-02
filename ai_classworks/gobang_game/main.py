from board import create_board, print_board, BLACK, WHITE
from game_rules import check_win, get_possible_moves
from ai import get_best_move
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=3, help='depth of the AI')
    parser.add_argument('--current_player', type=str, default='BLACK',
                        choices=['WHITE', 'BLACK'], help='BLACK: AI starts')
    args = parser.parse_args()
    return args


def play_game():
    """Run the game with AI vs. human."""
    board = create_board()
    args = parse_args()
    depth = args.depth
    current_player = args.current_player

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
                if len(board) > x >= 0 == board[x][y] and 0 <= y < len(board[0]):
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

if __name__ == "__main__":
    play_game()