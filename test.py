from plotters.line_plotter import LinesPlotter

plotter = LinesPlotter.load_data('results/' + '1556493818' + '/data.npy',
                                     ['reward', 'steps', 'end_state'])

plotter.