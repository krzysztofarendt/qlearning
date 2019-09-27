import random
import numpy as np


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

    def __init__(self, agent_id, position, board):
        self.id = agent_id
        self.position = np.array(position)
        self.epsilon = 0.2
        self.qtable = None
        self.board = board

    def init_qtable(self):
        # Action quality table
        qtable_shape = self.board.board.shape + (8, )  # (board size, board size, 8 moves)
        self.qtable = np.zeros(qtable_shape)

    def move(self, m):

        if isinstance(m, str):
            m = Agent.dir2mov[m]

        self.position = self.board.process_move(self.position, m)
        self.board.update()

    def explore(self):
        """
        Exploration: random move.
        """
        self.move(random.randint(0, 7))

    def exploit(self):
        """
        Exploitation: follow highest q-value.
        """
        pass

    def action(self):
        if random.uniform(0, 1) < self.epsilon:
            self.explore()
        else:
            self.exploit()
