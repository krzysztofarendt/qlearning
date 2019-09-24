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

    def move(self, m):
        # Move in one of 8 possible directions
        self.position += Agent.moves[m]


if __name__ == '__main__':

    agent = Agent(1, (1, 2))
    print(agent.position)

    agent.move(0)
    print(agent.position)