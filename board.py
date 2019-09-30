import os
import sys
import time
import random
import numpy as np
from scipy.spatial import distance
from agent import Agent


class Board:

    clear_console = 'clear' if os.name == 'posix' else 'CLS'

    goal_id = -1
    wall_id = -2

    # Rewards
    goal_reward = 100
    movement_reward = -1
    collision_reward = -5

    def __init__(self, size):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.board_hist = np.zeros((size, size), dtype=np.int32)
        self.agents = list()
        self.walls = list()
        self.action_str = '\n'
        self.goal_density = np.zeros((size, size))
        self.goal_not_achieved = True

    def set_goal(self, position):
        self.board[tuple(position)] = Board.goal_id

        # Calculate distance grid
        i_trg = position[0]
        j_trg = position[1]
        d = np.zeros(self.goal_density.shape)

        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                d[i, j] = distance.euclidean((i_trg, j_trg), (i, j))

        # Distance to density
        d_max = 8.
        d = d_max / (d + 0.1)  # To avoid zero division
        d[tuple(position)] = d_max

        self.goal_density = np.round(d, 0).astype(np.int16)

        self.log_action('Goal density:\n' 
            + str(self.goal_density.transpose()))

    def add_agent(self, position, agent=None):

        agent_added = False

        if self.board[position] == 0:
            agent_id = len(self.agents) + 1
            if agent is None:
                agent = Agent(agent_id, position, board=self)
            else:
                agent.position = np.array(position, dtype=np.int16)
                agent.board = self
                agent.id = agent_id
            self.agents.append(agent)
            self.board[position] = agent_id
            self.log_action(f'Agent {agent_id} added to {position}')
            agent_added = True
        else:
            self.log_action(f'Position {position} already taken! Agent not added!')

        return agent_added

    def take_off_agents(self):
        agents = self.agents
        self.agents = list()
        return agents 

    def process_move(self, position, move):
        """
        Check if move would result in a collision with another agent, wall, or board's edge.

        :param position: 2-element int array
        :param move: int (0..7)
        :return: 2-element int array
        """
        position = position
        requested_position = np.array(position, dtype=np.int16) + Agent.moves[move]
        reward = 0
        desc = None

        if (requested_position > self.size - 1).any() or (requested_position < 0).any():
            # Avoid going off board
            new_position = position
            reward += Board.collision_reward
            desc = 'Boundary collision'

        elif self.board[tuple(requested_position)] > 0:
            # Move blocked by another agent
            new_position = position
            reward += Board.collision_reward
            desc = 'Agent collision'

        elif tuple(requested_position) in self.walls:
            # Move blocked by a wall
            new_position = position
            reward += Board.collision_reward
            desc = 'Wall collision'

        elif self.board[tuple(requested_position)] == -1:
            # Goal achieved
            new_position = requested_position
            reward += Board.goal_reward
            desc = 'GOAL ACHIEVED!'
            self.take_off_goal(requested_position)
            self.goal_not_achieved = False

        else:
            # Move clear
            new_position = requested_position
            reward += Board.movement_reward

        # Add reward based on distance from goals
        reward += self.get_dist_reward(new_position)

        # Add position count to board history
        self.board_hist[tuple(new_position)] += 1

        return new_position, reward, desc

    def get_dist_reward(self, position):
        return self.goal_density[tuple(position)]

    def take_off_goal(self, position):
        self.goal_density = np.zeros(self.goal_density.shape)

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
