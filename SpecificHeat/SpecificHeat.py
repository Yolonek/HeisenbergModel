from HamiltonianClass import *
from CommonFunctions import print_and_store, make_directories
from time import time
import numpy as np
import matplotlib.pyplot as plt


class SpecificHeat(object):

    def __init__(self, J=1, L=0, delta=0, hamiltonian_reduced=False, is_pbc=False, temperature_range=None):
        self.J = J
        self.delta = delta
        self.L = L
        self.L_list = create_ascending_list(L)
        if hamiltonian_reduced:
            self.L_list = delete_odd_numbers(self.L_list)
        self.hamiltonian_reduced = hamiltonian_reduced
        self.pbc = is_pbc
        self.specific_heat_dict = {}
        self.json_file = self.file_name(extension='.json')
        if temperature_range is not None and J != 1:
            self.temperatures = temperature_range / J
        else:
            self.temperatures = temperature_range
        self.logs = ''

    def file_name(self, extension='.png'):
        name = f'SpecificHeat_L{self.L}_D{self.delta}'
        if self.hamiltonian_reduced:
            name += '_reduced'
        name += '_pbc' if self.pbc else '_obc'
        return name + extension

    def get_json_file_name(self):
        return self.json_file

    def figure_title(self):
        title = 'Graphs of $C_v(T)$ and $C_v(T) / L$'
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
                          'L': self.L_list,
                          'reduced': self.hamiltonian_reduced,
                          'periodic': self.pbc,
                          'temperatures': self.temperatures.tolist(),
                          'specific heat': {},
                          'logs': self.logs}
        for L, heat in self.specific_heat_dict.items():
            dict_with_data['specific heat'][L] = heat.tolist()
        save_json_file(dict_with_data, self.json_file, sub_dir=sub_dir)

    def load_data(self, sub_dir=''):
        dict_with_data = read_json_file(self.json_file, sub_dir=sub_dir)
        self.delta = dict_with_data['delta']
        self.L_list = dict_with_data['L']
        self.J = dict_with_data['J']
        self.hamiltonian_reduced = dict_with_data['reduced']
        self.pbc = dict_with_data['periodic']
        self.temperatures = np.array(dict_with_data['temperatures'])
        self.specific_heat_dict = {}
        self.logs = dict_with_data['logs']
        for L, heat in dict_with_data['specific heat'].items():
            self.specific_heat_dict[L] = np.array(heat)
        self.logs = dict_with_data['logs']

    def simulate_specific_heat(self):
        self.logs = print_and_store(self.logs)
        self.specific_heat_dict = {}
        for L in self.L_list:
            start_time_sim = time()
            quantum_state = QuantumState(L, self.J, self.delta, is_reduced=self.hamiltonian_reduced)
            self.logs = quantum_state.print_hamiltonian_data(return_msg=True)
            quantum_state.calculate_specific_heat_range(self.temperatures)
            self.specific_heat_dict[str(L)] = quantum_state.specific_heat
            stop_time_sim = time()
            self.logs = print_and_store(self.logs,
                                        message=f'Time elapsed: {round(stop_time_sim - start_time_sim, 3)} seconds')

    def plot_specific_heat(self):
        figure, axes = plt.subplots(1, 2, layout='constrained')
        for L in self.L_list:
            specific_heat = self.specific_heat_dict[str(L)]
            axes[0].plot(self.temperatures, specific_heat, label=f'L = {L}')
            axes[1].plot(self.temperatures, specific_heat / L, label=f'L = {L}')
        for index in range(2):
            axes[index].set_title(f'$\Delta = {self.delta}$')
            axes[index].legend(loc='upper right')
            axes[index].grid()
            axes[index].set(xlabel='T')
        axes[0].set(ylabel='$C_v$')
        figure.set_size_inches(6 * 2.56, 6 * 1.08)
        figure.suptitle(self.figure_title())
        return figure, axes


if __name__ == '__main__':
    J = 1
    L = 12
    delta = 1
    temperatures = linspace(0.1, 2, 100)
    periodic_boundary = False
    spin_zero = False

    specific_heat = SpecificHeat(J=J, L=L, delta=delta,
                                 hamiltonian_reduced=spin_zero,
                                 is_pbc=periodic_boundary,
                                 temperature_range=temperatures)

    results_path = 'results'
    image_path = 'images'
    make_directories([results_path, image_path])

    json_file_name = specific_heat.get_json_file_name()
    is_simulation_done = check_if_file_has_data(json_file_name, sub_dir=results_path)
    ask_to_redo_simulation = False

    if is_simulation_done:
        ask_to_redo_simulation = ask_to_replace_file()

    if is_simulation_done is False or ask_to_redo_simulation:
        start_time = time()
        specific_heat.simulate_specific_heat()
        stop_time = time()
        specific_heat.logs = print_and_store(specific_heat.logs,
                                             message=f'Program took {round(stop_time - start_time, 3)} seconds.')
        specific_heat.save_data(sub_dir=results_path)
    else:
        specific_heat.load_data(sub_dir=results_path)
        print(specific_heat.logs)

    figure, axes = specific_heat.plot_specific_heat()

    image_name = os.path.join(image_path, specific_heat.file_name())
    if not check_if_file_exists(image_name):
        figure.savefig(image_name)
