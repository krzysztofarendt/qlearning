import os
import sys
import time
import numpy as np
from agent import Agent


class Board:

    clear_console = 'clear' if os.name == 'posix' else 'CLS'

    goal_id = -1

    dir2mov = {
        'NW': 0, 'N': 1, 'NE': 2, 'E': 3,
        'SE': 4, 'S': 5, 'SW': 6, 'W': 7
    }

    def __init__(self, size):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.agents = list()
        self.action_str = ''

    def set_goal(self, position):
        self.board[position] = Board.goal_id

    def add_agent(self, position):
        if self.board[position] == 0:
            agent_id = len(self.agents) + 1
            self.agents.append(Agent(agent_id, position))
            self.board[position] = agent_id
            self.log_action(f'Agent {agent_id} added to {position}')
        else:
            self.log_action(f'Position {position} already taken! Agent not added!')

    def will_collide(self, position, move):
        """
        Check if move would result in a collision with another agent.

        :param position: tuple
        :param move: int (0..7)
        :return: bool
        """
        collide = False
        new_position = np.array(position) + Agent.moves[move]
        new_position = np.clip(new_position, 0, self.size - 1)

        if self.board[new_position[0], new_position[1]] > 0:
            # Will collide with other agent
            collide = True

        return collide

    def move_agent(self, agent_id, move):
        """
        :param agent_id: int
        :param move: str (e.g. 'NE') or int (0..7)
        :return: None
        """
        if isinstance(move, str):
            move_num = Board.dir2mov[move]
        else:
            move_num = move

        a = self.get_agent(agent_id)
        
        if not self.will_collide(a.position, move_num):
            a.move(move_num)
            # Do not move off board
            a.position = np.clip(a.position, 0, self.size - 1)
            self.update()
            self.log_action(f'Agent {agent_id} moves {move}')
        else:
            self.log_action(f'Agent {agent_id} will not move {move} to avoid collision!')

    def get_agent(self, id):
        return self.agents[id - 1]

    def get_pretty_view(self):
        """
        Return a rotated array such that (0, 0) is in the bottom left corner.
        Use it only for printing in terminal.
        """
        return np.flip(self.board, 1).transpose()

    def log_action(self, action):
        self.action_str += '\n' + action

    def pop_action(self):
        action = self.action_str
        self.action_str = '\n'
        return action

    def update(self):
        """
        Update agents positions on board.
        """
        self.board[self.board > 0] = 0

        for a in self.agents:
            self.board[a.position[0], a.position[1]] = a.id

    def show(self, sleep=0.5):
        """
        Print board on screen and flush action strings.

        :param sleep: Time in seconds
        :return: None
        """
        os.system(Board.clear_console)
        outstr = self.get_pretty_view().__str__()
        outstr = outstr.replace('-1', ' x')
        outstr = outstr.replace('0', '.')
        outstr += self.pop_action()
        outstr += '\n'
        sys.stdout.write(outstr)
        sys.stdout.flush()
        time.sleep(sleep)


if __name__ == '__main__':

    s = 0.5

    board = Board(10)
    board.set_goal(position=(4, 2))
    board.show(s)
    board.add_agent(position=(1, 0))  # Agent 1
    board.show(s)
    board.add_agent(position=(2, 3))  # Agent 2
    board.show(s)
    board.move_agent(agent_id=1, move='N')
    board.show(s)
    board.move_agent(agent_id=1, move='NW')
    board.show(s)
    board.move_agent(agent_id=2, move='S')
    board.show(s)
    board.move_agent(agent_id=1, move='E')
    board.show(s)
    board.move_agent(agent_id=1, move='E')
    board.show(s)
