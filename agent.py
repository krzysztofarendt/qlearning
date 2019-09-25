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

    def __init__(self, agent_id, position):
        self.id = agent_id
        self.position = np.array(position)
        self.prob_explore = 0.2
        self.qtable = None

    def init_tables(self, board):
        # Action quality table
        qtable_shape = board.board.shape + (8, )  # (board size, board size, 8 moves)
        self.qtable = np.zeros(qtable_shape)

        # Possible moves table
        mtable_shape = board.board.shape + (8, )  # (board size, board size, 8 moves)
        self.mtable = np.ones(mtable_shape)

    def move(self, m):
        # Move in one of 8 possible directions
        pos_t1 = tuple(self.position)
        self.position += Agent.moves[m]
        self.position = np.clip(self.position, 0, self.mtable.shape[0] - 1)
        pos_t2 = tuple(self.position)

        if pos_t2 == pos_t1:
            # Didn't move
            self.mtable[pos_t2] -= 0.1
        else:
            # Move successful - reset to 1
            self.mtable[pos_t2] = 1.

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
        if random.uniform(0, 1) < self.prob_explore:
            self.explore()
        else:
            self.exploit()


if __name__ == '__main__':

    agent = Agent(1, (1, 2))
    print(agent.position)

    agent.move(0)
    print(agent.position)