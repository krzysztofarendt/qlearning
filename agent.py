import random
import numpy as np
import pandas as pd


class Agent:

    moves = {
        #            dx  dy
        0: np.array((-1,  1)),  # NW
        1: np.array(( 0,  1)),  # N
        2: np.array(( 1,  1)),  # NE
        3: np.array(( 1,  0)),  # E
        4: np.array(( 1, -1)),  # SE
        5: np.array(( 0, -1)),  # S
        6: np.array((-1, -1)),  # SW
        7: np.array((-1,  0))   # W
    }

    dir2mov = {
        'NW': 0, 'N': 1, 'NE': 2, 'E': 3,
        'SE': 4, 'S': 5, 'SW': 6, 'W': 7
    }

    mov2dir = {v: k for k, v in dir2mov.items()}

    def __init__(self, agent_id, position, board, qtable=None):
        self.id = agent_id
        self.position = np.array(position, dtype=np.int16)
        self.qtable = None
        self.board = board

        # Learning parameters
        self.epsilon = 0.05
        self.alpha = 0.1
        self.gamma = 0.9

        # Initialize Q-table
        self.init_qtable()

    def init_qtable(self, path=None):
        if path is None:
            qtable_shape = self.board.board.shape + (8, )  # (board size, board size, 8 moves)
            self.qtable = np.zeros(qtable_shape)
        else:
            self.qtable = self.read_qtable(path)

    def move(self, m):

        old_position = self.position

        # If m is str, convert to int
        if isinstance(m, str):
            m = Agent.dir2mov[m]

        # Process move
        self.position, reward, desc = self.board.process_move(self.position, m)

        # Log feedback
        if desc is not None:
            self.board.log_action(f'Agent {self.id}: {desc}')

        # Re-render board
        self.board.update()

        # Update Q-table
        self.update_Qtable(old_position, m, self.position, reward)

    def update_Qtable(self, old_state, action, new_state, reward):
        self.qtable[old_state[0], old_state[1], action] = \
            (1 - self.alpha) * \
            self.qtable[old_state[0], old_state[1], action] + \
            self.alpha * \
            (
                reward +
                self.gamma * np.max(self.qtable[new_state[0], new_state[1], :])
            )

    def explore(self):
        """
        Exploration: random move.
        """
        self.move(random.randint(0, 7))

    def exploit(self):
        """
        Exploitation: follow highest q-value.
        """
        q_pos = self.qtable[self.position[0], self.position[1], :]
        best_move = q_pos[(q_pos == q_pos.max())]

        if best_move.size > 1:
            best_move = np.random.choice(best_move, size=1)[0]
        
        self.move(best_move)

    def action(self):
        if random.uniform(0, 1) < self.epsilon:
            self.explore()
        else:
            self.exploit()

    def save_qtable(self, path):
        np.save(path, self.qtable)

    def read_qtable(self, path):
        qtable = np.load(path)
        return qtable