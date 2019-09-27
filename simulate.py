import random
import numpy as np
from board import Board
from agent import Agent

# Frame length [s]
s = 0.1

# Build world
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

# Get agents
a1 = board.get_agent(1)
a2 = board.get_agent(2)

# Move agents randomly
for i in range(200):
    a1.move(random.randint(0, 7))
    board.show(s)
    a2.move(random.randint(0, 7))
    board.show(s)