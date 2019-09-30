import random
import time
import numpy as np
import pandas as pd
from board import Board
from agent import Agent

# Frame length [s]
s = 0.05

def build_world():

    board = Board(10)

    board.set_goal(position=(8, 4))
    # board.show(s)

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
    # board.show(1)

    return board


# Agent placeholders
a1 = None
# a2 = None

# Learning episodes
n_episodes = 1000
hist = pd.DataFrame(index=pd.Index(range(n_episodes), name='episode'),
    data={'steps': np.zeros(n_episodes)})

for episode in range(n_episodes):

    board = build_world()

    # Add agents
    agent_added = False
    while not agent_added:
        agent_added = board.add_agent(
            position=(np.random.randint(0, 7), np.random.randint(0, 7)),
            agent=a1
        )
    # board.add_agent(position=(2, 3), agent=a2)
    # board.show(s)

    try:
        a1 = board.get_agent(1)
        # a2 = board.get_agent(2)
    except IndexError as e:
        print(board.agents)
        raise e

    t = 0
    while board.goal_not_achieved:
        a1.move(random.randint(0, 7))
        # board.show(s)
        # a2.move(random.randint(0, 7))
        # board.show(s)

        t += 1

    # print(f'\nFinished in {t} steps...')
    # time.sleep(s)

    # Take off agents
    a1, = tuple(board.take_off_agents())
    # a1, a2 = tuple(board.take_off_agents())

    hist.loc[episode, 'steps'] = t

# Save Q-tables
a1.save_qtable(f'./qtables/agent1.npy')
# a2.save_qtable(f'./qtables/agent2.npy')

hist.to_csv('hist.csv')
np.save('board_hist.npy', board.board_hist)
