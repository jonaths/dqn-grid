import numpy as np
import matplotlib.pyplot as plt
from plotters.plotter import PolicyPlotter


def flip_states_per_rows(q_table, num_states, num_rows):
    """
    Invierte el orden de los estados cada num_rows en la tabla q
    :param q_table:
    :param num_states:
    :param num_rows:
    :return:
    """
    for i in range(0, num_states, num_rows):
        flipped = np.flip(q_table[i:i + num_rows, :], axis=0)
        q_table[i:i + num_rows, :] = flipped
    return q_table


def prepare_q_table(num_rows, num_cols, num_actions, agent):
    """
    Genera tabla q usando el modelo de agent para cada estado accion
    :param num_rows:
    :param num_cols:
    :param num_actions:
    :param agent:
    :return:
    """
    num_states = num_rows * num_cols

    # q_table = np.random.rand(num_states, num_actions)
    q_table = np.zeros(shape=(num_states, num_actions))

    # genera lista de estados
    for s in range(num_states):
        state_one_hot = np.eye(num_states)[s]
        # modifica el orden de los estados para generar politica
        q_table[s] = agent.online_q_values.eval(feed_dict={agent.X_state: [state_one_hot]})
    return q_table


def plot_policy(q_table, num_rows, num_cols, labels):
    """
    Genera un heatmap indicando la politica
    :param q_table:
    :param num_rows:
    :param num_cols:
    :param labels:
    :return:
    """

    q_table = flip_states_per_rows(q_table, num_rows * num_cols, num_rows)

    # instanciar
    # num_rows x num_cols debe ser igual a la longitud de q_table
    # la coordenada (0,0) es q_table[0], la (1,0) es q_table[1]
    plotter = PolicyPlotter(q_table, num_rows, num_cols)

    # se pueden obtener summaries (max, min, o avg)
    max_summary = plotter.summarize(op='max')
    print("INFO: max summary")
    print(max_summary)

    # o recuperar la politica calculando la accion maxima de cada estado
    # esto regresa el valor maximo y los indices de cada estado, si se pasan labels
    # regresa esos labels en lugar del indice numerico
    summary, indices = plotter.get_policy(labels=labels)
    print("INFO: policy")
    print(summary)
    print(indices)

    # tambien se puede generar el mapa de politicas
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    im, cbar, texts = plotter.build_policy(labels=labels)
    fig.tight_layout()
    fig.savefig('plots/policy.png', dpi=100)
    # plt.show()


def prepare_fear_model(num_rows, num_cols, num_actions, agent):
    num_states = num_rows * num_cols
    fear_model = np.zeros(shape=(num_states, num_actions, agent.k_bins+1))

    # genera lista de estados
    for s in range(num_states):
        state_one_hot = np.eye(num_states)[s]
        fear_model[s] = agent.get_state_actions(state_one_hot)
    fear_model_per_bin = np.transpose(fear_model, (2, 0, 1))
    return fear_model_per_bin


def plot_fear_models_per_bin(fear_model, num_rows, num_cols, labels, file_name='fear_model'):

    fear_model = flip_states_per_rows(fear_model, num_rows * num_cols, num_rows)

    plotter = PolicyPlotter(fear_model, num_rows, num_cols)

    # tambien se puede generar el mapa de politicas
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    im, cbar, texts = plotter.build_policy(labels=labels)
    fig.tight_layout()
    fig.savefig('plots/'+file_name+'.png', dpi=100)





