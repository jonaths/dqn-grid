# 1556642255_lmb-0_k-05

import matplotlib.pyplot as plt
from plotters.line_plotter import LinesPlotter
from collections import Counter

output_folder = 'plots/'
exp_name = 'exp_name'
danger_states = [0, 70]

# experiments_list = [
#     'rev01/cartpole/04-no-rms', 'rev01/cartpole/05-rms-in2',
#     'rev01/cartpole/06-rms-in4', 'rev02/06-sarsa'

experiments_list = [
    '1556642255_lmb-0_k-05'
]
labels = ['No RMS']
lines = ['-', '--', ':', '-.']

fig, ax = plt.subplots()

for i in range(len(experiments_list)):
    plotter = LinesPlotter.load_data('results/' + experiments_list[i] + '/data.npy',
                                     ['reward', 'steps', 'end_state'])

    print(plotter.data.shape)
    end_states = plotter.data[0, :, 2]
    print(Counter(end_states).keys())
    print(Counter(end_states).values())

