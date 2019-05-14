# 1556642255_lmb-0_k-05

import matplotlib.pyplot as plt
from plotters.line_plotter import LinesPlotter
import numpy as np
output_folder = 'plots/'
exp_name = 'exp_name'

# para varios experimentos en un solo data.npy
experiments_list = [
    '1556826137-01',
    '1556829021-02',
    '1556831512-03'
]

merged = np.zeros(shape=(len(experiments_list), 3000, 3))

labels = ['No RMS']
lines = ['-', '--', ':', '-.']

fig, ax = plt.subplots()

for i in range(len(experiments_list)):
    plotter = LinesPlotter.load_data('results/' + experiments_list[i] + '/data.npy',
                                     ['reward', 'steps', 'end_state'])
    print(merged[i].shape, plotter.data.shape)
    merged[i] = plotter.data[0]


print(merged.shape)

np.save('data.npy', merged)