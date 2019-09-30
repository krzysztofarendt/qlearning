#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from agent import Agent

#%% Q-table
# Plot max Q for each position
q = np.load('qtables/agent1.npy')
q = np.max(q, axis=2)

fig, ax = plt.subplots(1, 1)
sns.heatmap(np.flip(q, 1).transpose(), annot=True)
plt.show()

#%% Best move
q = np.load('qtables/agent1.npy')
m = np.argmax(q, 2)
m = np.flip(m, 1).transpose()
d = np.full(m.shape, '  ', dtype='<U2')
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        d[i, j] = Agent.mov2dir[m[i, j]].rjust(2)

print(d)
print(m)

#%% Position heat map
# Plot number of times each tile was visited
bhist = np.load('board_hist.npy')
fig, ax = plt.subplots(1, 1)
sns.heatmap(np.flip(bhist, 1).transpose(), annot=True)
ax.set_title(f'sum={bhist.sum()}')
plt.show()

#%% Learning history
# Number of steps required to reach the goal
hist = pd.read_csv('hist.csv')
hist = hist.rolling(25).mean()
plt.plot(hist['steps'])
plt.show()



#%%
