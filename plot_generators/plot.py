import matplotlib.pyplot as plt
import matplotlib
from plotters.line_plotter import LinesPlotter

output_folder = ''
exp_name = 'grid'
danger_states = [15, 19, 23, 27]

experiments_list = [
    'test'
]

labels = ['No RMS']
lines = ['-']

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)

for i in range(len(experiments_list)):

    plotter = LinesPlotter.load_data(experiments_list[i] + '/data.npy',
                                     ['reward', 'steps', 'end_state'], num_episodes=1000)

    print(plotter.data.shape)

    fig, ax = plotter.get_var_line_plot(
        ['reward'], func='average', linestyle=lines[i], fig=fig, ax=ax, label=labels[i], window_size=120)

fig.legend(loc=10)
plt.xlabel('Episodes')
plt.ylabel('Failure count')
plt.tight_layout()
plt.savefig(output_folder + 'reward-' + exp_name + '.png', dpi=300)

# get_var_line_plot(self, var_name_list, func, linestyle=None, window_size=20, fig=None,
#                           ax=None, label=None):