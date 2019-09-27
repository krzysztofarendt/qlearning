import os
import sys
import time
import random
import numpy as np
from agent import Agent


class Board:

    clear_console = 'clear' if os.name == 'posix' else 'CLS'

    goal_id = -1
    wall_id = -2

    dir2mov = {
        'NW': 0, 'N': 1, 'NE': 2, 'E': 3,
        'SE': 4, 'S': 5, 'SW': 6, 'W': 7
    }

    mov2dir = {v: k for k, v in dir2mov.items()}

    def __init__(self, size):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.agents = list()
        self.walls = list()
        self.action_str = '\n'

    def set_goal(self, position):
        self.board[position] = Board.goal_id

    def add_agent(self, position):
        if self.board[position] == 0:
            agent_id = len(self.agents) + 1
            agent = Agent(agent_id, position, board=self)
            agent.init_qtable()
            self.agents.append(agent)
            self.board[position] = agent_id
            self.log_action(f'Agent {agent_id} added to {position}')
        else:
            self.log_action(f'Position {position} already taken! Agent not added!')

    def process_move(self, position, move):
        """
        Check if move would result in a collision with another agent, wall, or board's edge.

        :param position: 2-element int array
        :param move: int (0..7)
        :return: 2-element int array
        """
        position = position
        requested_position = np.array(position) + Agent.moves[move]

        if (requested_position > self.size - 1).any() or (requested_position < 0).any():
            # Avoid going off board
            new_position = position

        elif self.board[tuple(requested_position)] > 0:
            # Move blocked by another agent
            new_position = position

        elif tuple(requested_position) in self.walls:
            # Move blocked by a wall
            new_position = position

        else:
            # Move clear
            new_position = requested_position

        return new_position                 

    def get_agent(self, id):
        return self.agents[id - 1]

    def add_wall(self, position):
        self.board[position] = Board.wall_id
        self.walls.append(position)
        self.log_action(f'Wall added at {position}')

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
        outstr = outstr.replace('-2', ' #')
        outstr = outstr.replace('0', '.')
        outstr = outstr.replace('[[', ' [')
        outstr = outstr.replace(']]', '] ')
        outstr = outstr.replace('[', ' ')
        outstr = outstr.replace(']', ' ')
        outstr += self.pop_action()
        outstr += '\n'
        sys.stdout.write(outstr)
        sys.stdout.flush()
        time.sleep(sleep)
