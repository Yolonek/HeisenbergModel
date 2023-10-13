from HamiltonianClass import *
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class Perturbation(object):
    def __init__(self, J=1, L=0, delta=0, h_list=None, hamiltonian_reduced=False, is_pbc=False, omega_bin=0.05,
                 time_range=None):
        self.J = J
        self.delta = delta
        if hamiltonian_reduced:
            self.L = L if L % 2 == 0 else L - 1
        else:
            self.L = L
        self.hamiltonian_reduced = hamiltonian_reduced
        self.pbc = is_pbc
        self.k_list = [k for k in range(L + 1)]
        self.wave_vector_list = [wave_vector(L, k) for k in self.k_list]
        self.omega_bin = omega_bin
        self.h_list = h_list
        self.linear_response_dict = {}
        self.time_range = time_range
        self.omega_range = None
        self.json_file = self.file_name(extension='.json')

    def file_name(self, extension='.png'):
        name = f'Perturbation_L{self.L}'
        if self.hamiltonian_reduced:
            name += '_reduced'
        name += '_pbc' if self.pbc else '_obc'
        return name + extension

    def get_json_file_name(self):
        return self.json_file

    def figure_title(self):
        title = f'Linear response in function of omega and wave vector, size L={self.L}'
        if self.hamiltonian_reduced:
            title += ', total spin = 0'
        else:
            title += ', full-size hamiltonian'
        if self.pbc:
            title += ', periodic boundary conditions'
        else:
            title += ', open boundary conditions'
        return title

    def save_data(self):
        dict_with_data = {'J': self.J,
                          'delta': self.delta,
                          'L': self.L,
                          'h': self.h_list,
                          'wave vector': self.wave_vector_list,
                          'omega': (self.omega_range.tolist(), self.omega_bin),
                          'time': self.time_range.tolist(),
                          'linear response': {}}
        for h_param, response in self.linear_response_dict.items():
            dict_with_data['linear response'][h_param] = response.tolist()
        save_json_file(dict_with_data, self.json_file)

    def load_data(self):
        dict_with_data = read_json_file(self.json_file)
        self.J = dict_with_data['J']
        self.delta = dict_with_data['delta']
        self.L = dict_with_data['L']
        self.h_list = dict_with_data['h']
        self.wave_vector_list = dict_with_data['wave vector']
        self.omega_range = np.array(dict_with_data['omega'][0])
        self.omega_bin = dict_with_data['omega'][1]
        self.time_range = np.array(dict_with_data['time'])
        for h_param, response in dict_with_data['linear response'].items():
            self.linear_response_dict[h_param] = np.array(response)

    def simulate_perturbation(self):
        start_time = time()
        start_time_sim = time()
        print('Preparing data for simulation...')
        self.linear_response_dict = {}
        quantum_state = QuantumState(self.L, self.J, self.delta,
                                     is_pbc=self.pbc,
                                     is_reduced=self.hamiltonian_reduced)
        self.omega_range = quantum_state.generate_linspace_of_omega(self.omega_bin, boundary=(0, 3))
        time_range_imaginary = convert_array_to_imaginary_part(self.time_range)
        stop_time_sim = time()
        print(f'Done! Time elapsed: {round(stop_time_sim - start_time_sim, 3)} seconds ')

        for h_param in self.h_list:
            linear_response_grid = None
            for wave_vec, k in zip(self.wave_vector_list, self.k_list):
                start_time_sim = time()
                perturbation = quantum_state.set_wave_vector_perturbation(h_param, wave_vec)
                quantum_state.eigenstates(add_perturbation=perturbation)
                quantum_state.set_vector_from_eigenstate(0)
                quantum_state.eigenstates()
                time_evo_dict = quantum_state.spin_evolution_range(self.time_range)
                linear_response = quantum_state.calculate_linear_response_fft(self.omega_range, time_range_imaginary,
                                                                              time_evo_dict, wave_vec)[:, None]
                if k == 0:
                    linear_response_grid = linear_response
                else:
                    linear_response_grid = np.concatenate((linear_response_grid, linear_response), axis=1)
                stop_time_sim = time()
                print(f'h = {h_param}, k = {k}, '
                      f'time elapsed: {round(stop_time_sim - start_time_sim, 3)} seconds')
            self.linear_response_dict[str(h_param)] = linear_response_grid
        stop_time = time()
        print(f'Simulation took {round(stop_time - start_time, 3)} seconds.')

    def plot_perturbation(self):
        amount_of_plots = len(self.h_list)
        wave_vector_range = np.array(self.wave_vector_list)
        figure, axis = plt.subplots(1, amount_of_plots)
        colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list('rg', colors, N=256)
        for h_param, index in zip(self.linear_response_dict, range(amount_of_plots)):
            linear_response_grid = np.sqrt(self.linear_response_dict[str(h_param)])
            # linear_response_grid = self.linear_response_dict[str(h_param)]
            graph_title = f'L = {self.L}, h = {h_param}, delta = {self.delta}, J = {self.J}'
            axis[index].set_title(graph_title)
            sq_min, sq_max = 0, int(linear_response_grid.max())
            sq_max = 1 if sq_max == 0 else sq_max
            grid = axis[index].pcolormesh(wave_vector_range, self.omega_range, linear_response_grid, cmap=cmap,
                                          vmin=sq_min, vmax=sq_max, shading='gouraud')
            figure.colorbar(grid, ax=axis[index])
            axis[int(amount_of_plots / 2)].set(xlabel='wave vector')
        axis[0].set(ylabel='omega')
        figure.suptitle(self.figure_title())
        figure.set_size_inches(8 * 2.56, 8 * 1.08)
        plt.tight_layout()
        return figure, axis


# Parameters
J = 1
delta = 1
L = 6
h = [0.01, 1, 10]
periodic_boundary = True
spin_zero = True
omega_bin = 0.05
q = wave_vector(L, 2)
times = linspace(0, 100, 50)

start_time = time()

perturbation = Perturbation(J=J, L=L, delta=delta,
                            h_list=h,
                            hamiltonian_reduced=spin_zero,
                            is_pbc=periodic_boundary,
                            omega_bin=omega_bin,
                            time_range=times)

json_file_name = perturbation.get_json_file_name()
is_simulation_done = check_if_file_has_data(json_file_name)
ask_to_redo_simulation = False

if is_simulation_done:
    ask_to_redo_simulation = ask_to_replace_file()

if is_simulation_done is False or ask_to_redo_simulation:
    perturbation.simulate_perturbation()
    perturbation.save_data()
else:
    perturbation.load_data()

stop_time = time()
print(f'Program took {round(stop_time - start_time, 3)} seconds.')

fig1, ax1 = perturbation.plot_perturbation()

fig1.savefig(perturbation.file_name())
