import matplotlib.pyplot as plt
from plotters.line_plotter import LinesPlotter

output_folder = 'plots/'
exp_name = 'exp_name'
danger_states = [0, 70]

# experiments_list = [
#     'rev01/cartpole/04-no-rms', 'rev01/cartpole/05-rms-in2',
#     'rev01/cartpole/06-rms-in4', 'rev02/06-sarsa'

experiments_list = [
    '1556586291_lmb-0'
]
labels = ['No RMS']
lines = ['-', '--', ':', '-.']

fig, ax = plt.subplots()

for i in range(len(experiments_list)):
    plotter = LinesPlotter.load_data('results/' + experiments_list[i] + '/data.npy',
                                     ['reward', 'steps', 'end_state'])

    # fig, ax = plotter.get_var_line_plot(
    #     ['steps'], func='average', linestyle=lines[i], fig=fig, ax=ax, label=labels[i],
    #     window_size=120)

    fig, ax = plotter.get_var_cummulative_matching_plot(
        'end_state', danger_states, linestyle=lines[i], fig=fig, ax=ax, label=labels[i])

fig.legend()
plt.tight_layout()
plt.savefig('end-cummulative-' + exp_name + '.png')