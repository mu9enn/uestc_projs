import numpy as np
import random

def agent1(obs, config, model):
    # Reshape the board to match the model's expected input shape (1,6,7)
    board = np.array(obs['board']).reshape(1, config.rows, config.columns)
    col, _ = model.predict(board)
    # Validate the move: check that the chosen column is valid.
    is_valid = (obs['board'][int(col)] == 0)
    if is_valid:
        return int(col)
    else:
        valid_moves = [c for c in range(config.columns) if obs['board'][c] == 0]
        return random.choice(valid_moves)
