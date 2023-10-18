from HamiltonianClass import *
from CommonFunctions import print_and_store, make_directories
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class LinearResponse(object):

    def __init__(self, J=1, L=0, delta=0, temperature_list=None, hamiltonian_reduced=False, is_pbc=False, omega_bin=0.1):
        self.J = J
        self.delta = delta
        if hamiltonian_reduced:
            self.L = L if L % 2 == 0 else L - 1
        else:
            self.L = L
        self.hamiltonian_reduced = hamiltonian_reduced
        self.k_list = [k for k in range(L + 1)]
        self.wave_vector_list = [wave_vector(L, k) for k in self.k_list]
        self.pbc = is_pbc
        self.linear_response_dict = {}
        self.omega_bin = omega_bin
        self.temperatures = temperature_list
        self.omega_range = None
        self.json_file = self.file_name(extension='.json')
        self.logs = ''

    def file_name(self, extension='.png'):
        name = f'LinearResponse_L{self.L}_D{self.delta}T-'
        for T in self.temperatures:
            name += f'{T}-'
        name += 'pbc' if self.pbc else 'obc'
        if self.hamiltonian_reduced:
            name += '_reduced'
        return name + extension

    def get_json_file_name(self):
        return self.json_file

    def figure_title(self):
        title = f'Linear response $S(q, \omega)$, size $L={self.L}$'
        if self.hamiltonian_reduced:
            title += r', $\hat{S}^z_{tot} = 0$'
        else:
            title += ', full-size hamiltonian'
        if self.pbc:
            title += ', periodic boundary conditions'
        else:
            title += ', open boundary conditions'
        return title

    def save_data(self, sub_dir=''):
        dict_with_data = {'J': self.J,
                          'delta': self.delta,
                          'L': self.L,
                          'temperatures': self.temperatures,
                          'wave vector': self.wave_vector_list,
                          'omega': (self.omega_range.tolist(), self.omega_bin),
                          'linear response': {},
                          'logs': self.logs}
        for temperature, response in self.linear_response_dict.items():
            dict_with_data['linear response'][temperature] = response.tolist()
        save_json_file(dict_with_data, self.json_file, sub_dir=sub_dir)

    def load_data(self, sub_dir=''):
        dict_with_data = read_json_file(self.json_file, sub_dir=sub_dir)
        self.J = dict_with_data['J']
        self.delta = dict_with_data['delta']
        self.L = dict_with_data['L']
        self.temperatures = dict_with_data['temperatures']
        self.wave_vector_list = dict_with_data['wave vector']
        self.omega_range = np.array(dict_with_data['omega'][0])
        self.omega_bin = dict_with_data['omega'][1]
        self.logs = dict_with_data['logs']
        for temperature, response in dict_with_data['linear response'].items():
            self.linear_response_dict[temperature] = np.array(response)

    def simulate_linear_response(self):
        self.logs = print_and_store(self.logs)
        self.linear_response_dict = {}
        quantum_state = QuantumState(self.L, self.J, self.delta,
                                     is_pbc=self.pbc,
                                     is_reduced=self.hamiltonian_reduced)
        self.logs = quantum_state.print_hamiltonian_data(return_msg=True)
        self.omega_range = quantum_state.generate_linspace_of_omega(self.omega_bin)
        for temperature in self.temperatures:
            linear_response_grid = None
            for wave_vector, k in zip(self.wave_vector_list, self.k_list):
                start_time_sim = time()
                quantum_state.set_sq_operator(wave_vector)
                if k == 0:
                    linear_response_grid = quantum_state.calculate_linear_response(self.omega_range,
                                                                                   self.omega_bin,
                                                                                   temperature)[:, None]
                else:
                    linear_response = quantum_state.calculate_linear_response(self.omega_range,
                                                                              self.omega_bin,
                                                                              temperature)[:, None]
                    linear_response_grid = np.concatenate((linear_response_grid, linear_response), axis=1)
                stop_time_sim = time()
                print_and_store(self.logs,
                                message=f'Temperature = {temperature}, k = {k}, '
                                        f'time elapsed: {round(stop_time_sim - start_time_sim, 3)} seconds')
            self.linear_response_dict[str(temperature)] = linear_response_grid

    def plot_linear_response(self):
        amount_of_plots = len(self.temperatures)
        wave_vector_range = np.array(self.wave_vector_list)
        figure, axes = plt.subplots(1, amount_of_plots, layout='constrained')
        cmap = LinearSegmentedColormap.from_list('rg', ["w", "r"], N=256)
        for index, temperature in enumerate(self.linear_response_dict):
            linear_response_grid = self.linear_response_dict[temperature]
            graph_title = f'$L = {self.L}$, $T = {temperature}$, $\Delta = {self.delta}$'
            axes[index].set_title(graph_title)
            sq_min, sq_max = 0, int(linear_response_grid.max())
            sq_max = 1 if sq_max == 0 else sq_max
            grid = axes[index].pcolormesh(wave_vector_range, self.omega_range, linear_response_grid, cmap=cmap,
                                          vmin=sq_min, vmax=sq_max)
            figure.colorbar(grid, ax=axes[index])
            axes[int(amount_of_plots / 2)].set(xlabel='wave vector $q$')
        axes[0].set(ylabel='$\omega$')
        figure.suptitle(self.figure_title())
        figure.set_size_inches(8 * 2.56, 8 * 1.08)
        return figure, axes


if __name__ == '__main__':
    J = 1
    delta = 1
    L = 10
    temperature_range = [0, 1, float('inf')]

    periodic_boundary = False
    spin_zero = True
    omega_bin = 0.1

    linear_response = LinearResponse(J=J, L=L, delta=delta,
                                     hamiltonian_reduced=spin_zero,
                                     is_pbc=periodic_boundary,
                                     temperature_list=temperature_range,
                                     omega_bin=omega_bin)

    results_path = 'results'
    image_path = 'images'
    make_directories([results_path, image_path])

    json_file_name = linear_response.get_json_file_name()
    is_simulation_done = check_if_file_has_data(json_file_name, sub_dir=results_path)
    ask_to_redo_simulation = False

    if is_simulation_done:
        ask_to_redo_simulation = ask_to_replace_file()

    if is_simulation_done is False or ask_to_redo_simulation:
        start_time = time()
        linear_response.simulate_linear_response()
        stop_time = time()
        linear_response.logs = print_and_store(linear_response.logs,
                                               message=f'Program took {round(stop_time - start_time, 3)} seconds.')
        linear_response.save_data(sub_dir=results_path)
    else:
        linear_response.load_data(sub_dir=results_path)
        print(linear_response.logs)

    figure, axes = linear_response.plot_linear_response()

    image_name = os.path.join(image_path, linear_response.file_name())
    if not check_if_file_exists(image_name):
        figure.savefig(image_name)
