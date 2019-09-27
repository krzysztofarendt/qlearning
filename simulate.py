import random
import time
import numpy as np
from board import Board
from agent import Agent

# Frame length [s]
s = 0.001

def build_world():

    board = Board(10)

    board.set_goal(position=(8, 4))
    board.show(4)

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
    board.show(1)

    return board


# Agent placeholders
a1 = None
a2 = None

# Learning episodes
n_episodes = 10

for episode in range(n_episodes):

    board = build_world()

    # Add agents
    board.add_agent(position=(3, 1), agent=a1)
    board.add_agent(position=(2, 3), agent=a2)
    board.show(1)

    a1 = board.get_agent(1)
    a2 = board.get_agent(2)

    t = 0
    while board.goal_not_achieved:
        a1.move(random.randint(0, 7))
        board.show(s)
        a2.move(random.randint(0, 7))
        board.show(s)

        t += 1

    print(f'\nFinished in {t} steps...')
    time.sleep(5)

    # Take off agents
    a1, a2 = tuple(board.take_off_agents())

    # Save Q-tables
    a1.save_qtable(f'./qtables/agent1_{episode}.npy')
    a2.save_qtable(f'./qtables/agent2_{episode}.npy')
