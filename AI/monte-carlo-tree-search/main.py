import numpy as np
from node import TwoPlayersGameMonteCarloTreeSearchNode
from search import MonteCarloTreeSearch
from tic_tac_toe import TicTacToeGameState

state = np.zeros((3,3))
initial_board_state = TicTacToeGameState(state = state, next_to_move=1)

root = TwoPlayersGameMonteCarloTreeSearchNode(state = initial_board_state)
mcts = MonteCarloTreeSearch(root)
best_node = mcts.best_action(10000)