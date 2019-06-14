from keras.models import load_model
import numpy as np
import keras
import math

model = load_model('best_model_5_64_16.h5')
model._make_predict_function()

def convertMovesToInput(prevmoves):
    board = np.full((7, 6), 0)
    
    # Iterate over the moves and fill the board with the corresponding piece
    for i, c in enumerate(prevmoves):
        color = i % 2 + 1
        cIdx = int(c) - 1
        emptyIdx = np.where(board[cIdx] == 0)[0][0]
        board[cIdx, emptyIdx] = color
    
    # rotate the board so that it is more natural to us humans
    board = np.rot90(board)
    
    # split the board out into two distinct planes, one for black and one for red.
    black = np.copy(board)
    red = np.copy(board)

    black[black == 2] = 0
    red[red == 1] = 0
    red[red == 2] = 1
    
    # The first plane should be the next player to move
    planes = [black, red]
    if (len(prevmoves) != 0 and len(prevmoves) % 2 == 1):
        planes.reverse()

    # stack the 2D planes into a 3D set of planes.
    planes = np.stack(planes)
    planes = np.rollaxis(planes, 0, 3)
    planes = planes.reshape(6, 7, 2)
    return planes

def getValidMoves(prev_moves):
    # Split input into an array and convert each character to an int
    moves = [int(move) for move in prev_moves]

    # Check each column to see if it is full of pieces
    valid_moves_flag = np.array([moves.count(move) < 6 for move in np.arange(1,8)])
    return np.arange(1,8)[valid_moves_flag].astype(np.str)

def getBestMove(prev_moves):
    print("Starting")
    valid_moves = np.char.array(getValidMoves(prev_moves))
    candidate_moves = prev_moves + valid_moves
    model_inputs = np.array([convertMovesToInput(candidate_state) for candidate_state in candidate_moves])
    predictions = model.predict(model_inputs)

    has_winning_predictions = np.array(predictions[:,1] > predictions[:,0]).any()
    best_winning_move = valid_moves[predictions[:,1].argmax()]
    best_drawing_move = valid_moves[predictions[:,2].argmax()]

    print(best_drawing_move)

    return best_winning_move if has_winning_predictions else best_drawing_move

def test(move):
    print("MOVING")
    return move

import os
print(os.getcwd())

