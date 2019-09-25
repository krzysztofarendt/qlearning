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
            agent = Agent(agent_id, position)
            agent.init_tables(self)
            self.agents.append(agent)
            self.board[position] = agent_id
            self.log_action(f'Agent {agent_id} added to {position}')
        else:
            self.log_action(f'Position {position} already taken! Agent not added!')

    def will_collide(self, position, move):
        """
        Check if move would result in a collision with another agent or wall.

        :param position: tuple
        :param move: int (0..7)
        :return: bool
        """
        collide = False
        new_position = np.array(position) + Agent.moves[move]
        new_position = np.clip(new_position, 0, self.size - 1)

        if self.board[new_position[0], new_position[1]] > 0:
            # Will collide with another agent or an edge of the board
            # If it wants to move off board, the value is equal to agent_id (>0),
            # because the position was clipped few lines above
            collide = True
        elif tuple(new_position) in self.walls:
            # Will collide with a wall
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
        
        if self.will_collide(a.position, move_num):
            self.log_action(f'Agent {agent_id} will not move {move} to avoid collision!')
        else:
            a.move(move_num)
            # Clip position to board dimensions - never move off board
            a.position = np.clip(a.position, 0, self.size - 1)
            self.update()
            self.log_action(f'Agent {agent_id} moves {move}')            

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


if __name__ == '__main__':

    s = 0.1

    board = Board(10)
    board.set_goal(position=(9, 0))
    board.set_goal(position=(6, 4))
    board.set_goal(position=(7, 4))
    board.set_goal(position=(8, 4))
    board.add_wall(position=(4, 2))
    board.add_wall(position=(4, 3))
    board.add_wall(position=(4, 4))
    board.add_wall(position=(4, 5))
    board.add_wall(position=(4, 6))
    board.add_wall(position=(5, 6))
    board.add_wall(position=(6, 6))
    board.add_wall(position=(7, 6))
    board.add_wall(position=(5, 2))
    board.add_wall(position=(6, 2))
    board.add_wall(position=(7, 2))
    board.add_wall(position=(8, 2))
    board.add_wall(position=(9, 2))
    board.show(s)
    board.add_agent(position=(3, 1))  # Agent 1
    board.show(s)
    board.add_agent(position=(2, 3))  # Agent 2
    board.show(s)

    # Random moves
    for i in range(100):
        mov2dir = board.mov2dir
        board.move_agent(agent_id=1, move=mov2dir[random.randint(0, 7)])
        board.move_agent(agent_id=2, move=mov2dir[random.randint(0, 7)])
        board.show(s)
    
    # Print mtables
    a1 = board.get_agent(1)
    print(a1.mtable)
