import numpy as np
import matplotlib.pyplot as plt
from plotters.plotter import PolicyPlotter

def main():

    num_actions = 4
    num_states = 20

    # la entrada es una tabla con los estados en las filas y  las acciones en las columnas
    q_table = np.random.rand(num_states, num_actions)

    # instanciar
    # num_rows x num_cols debe ser igual a la longitud de q_table
    # la coordenada (0,0) es q_table[0], la (1,0) es q_table[1]
    plotter = PolicyPlotter(q_table, num_rows=4, num_cols=5)

    # se pueden obtener summaries (max, min, o avg)
    max_summary = plotter.summarize(op='max')
    print("INFO: max summary")
    print(max_summary)

    # o recuperar la politica calculando la accion maxima de cada estado
    # esto regresa el valor maximo y los indices de cada estado, si se pasan labels
    # regresa esos labels en lugar del indice numerico
    summary, indices = plotter.get_policy(labels=['a', 'b', 'c', 'd'])
    print("INFO: policy")
    print(summary)
    print(indices)

    # tambien se puede generar el mapa de politicas
    fig, ax = plt.subplots()
    im, cbar, texts = plotter.build_policy(labels=['a', 'b', 'c', 'd'])
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()



