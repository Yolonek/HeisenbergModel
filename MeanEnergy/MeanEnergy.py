from HamiltonianClass import *
from CommonFunctions import print_and_store, make_directories
from time import time
import numpy as np
import matplotlib.pyplot as plt


class MeanEnergy(object):

    def __init__(self, J=1, L=0, deltas=None, temperature_range=None, hamiltonian_reduced=False, is_pbc=False, divided_by_L=False):
        self.J = J
        self.deltas = deltas
        self.L = L
        self.L_list = create_ascending_list(L)
        if hamiltonian_reduced:
            self.L_list = delete_odd_numbers(self.L_list)
        self.hamiltonian_reduced = hamiltonian_reduced
        self.pbc = is_pbc
        self.divided_by_L = divided_by_L
        self.mean_energy_dict = {}
        self.json_file = self.file_name(extension='.json')
        self.temperatures = temperature_range
        self.logs = ''

    def file_name(self, extension='.png'):
        name = f'MeanEnergy_L{self.L}D-'
        for delta in self.deltas:
            name += f'{delta}-'
        name += '_pbc' if self.pbc else '_obc'
        if self.divided_by_L:
            name += '_divided_by_L'
        if self.hamiltonian_reduced:
            name += '_reduced'
        return name + extension

    def get_json_file_name(self):
        return self.json_file

    def figure_title(self):
        title = 'Graph of mean energy for each delta'
        if self.divided_by_L:
            title += ', divided by L'
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
                          'deltas': self.deltas,
                          'L list': self.L_list,
                          'L': self.L,
                          'reduced': self.hamiltonian_reduced,
                          'periodic': self.pbc,
                          'divided': self.divided_by_L,
                          'temperatures': self.temperatures.tolist(),
                          'mean energy': {},
                          'logs': self.logs}
        for delta in self.mean_energy_dict:
            dict_with_data['mean energy'][delta] = {}
            for L, energy in self.mean_energy_dict[delta].items():
                dict_with_data['mean energy'][delta][L] = energy.tolist()
        save_json_file(dict_with_data, self.json_file, sub_dir=sub_dir)

    def load_data(self, sub_dir=''):
        dict_with_data = read_json_file(self.json_file)
        self.J = dict_with_data['J']
        self.deltas = dict_with_data['deltas']
        self.L = dict_with_data['L']
        self.L_list = dict_with_data['L list']
        self.hamiltonian_reduced = dict_with_data['reduced']
        self.pbc = dict_with_data['periodic']
        self.divided_by_L = dict_with_data['divided']
        self.temperatures = np.array(dict_with_data['temperatures'])
        for delta in dict_with_data['mean energy']:
            self.mean_energy_dict[delta] = {}
            for L, energy in dict_with_data['mean energy'][delta].items():
                self.mean_energy_dict[delta][L] = np.array(energy)
        self.logs = dict_with_data['logs']

    def simulate_mean_energy(self):
        self.logs = print_and_store(self.logs)
        self.mean_energy_dict = {}
        for delta in self.deltas:
            self.mean_energy_dict[str(delta)] = {}
            for L in self.L_list:
                start_time_sim = time()
                quantum_state = QuantumState(L, self.J, delta, is_reduced=self.hamiltonian_reduced)
                self.logs = quantum_state.print_hamiltonian_data(return_msg=True)
                quantum_state.calculate_mean_energy_range(self.temperatures)
                mean_energy = quantum_state.mean_energy
                if self.divided_by_L:
                    mean_energy = mean_energy / L
                self.mean_energy_dict[str(delta)][str(L)] = mean_energy
                stop_time_sim = time()
                self.logs = print_and_store(self.logs,
                                            message=f'Time elapsed: {round(stop_time_sim - start_time_sim, 3)} seconds')

    def plot_mean_energy(self):
        figure, axis = plt.subplots(1, len(self.deltas))
        for index, delta in enumerate(self.deltas):
            axis[index].set_title(f'$\Delta = {delta}$')
            for L in self.L_list:
                mean_energy = self.mean_energy_dict[str(delta)][str(L)]
                axis[index].plot(self.temperatures, mean_energy, label=f'L={L}')
            axis[index].grid()
            axis[index].legend(loc='lower right')
        axis[0].set(ylabel=r'$\langle E\rangle$')
        figure.supxlabel('Temperature')
        figure.set_size_inches(5.5 * 2.56, 5.5 * 1.08)
        figure.suptitle(self.figure_title())
        return figure, axis


if __name__ == '__main__':
    J = 1
    delta = [0.5, 1.0, 2.0]
    L = 12
    temperatures = linspace(0.1, 5, 300)

    periodic_boundary = True
    spin_zero = True
    is_divided_by_L = True

    mean_energy = MeanEnergy(J=J, L=L, deltas=delta,
                             temperature_range=temperatures,
                             hamiltonian_reduced=spin_zero,
                             is_pbc=periodic_boundary,
                             divided_by_L=is_divided_by_L)

    results_path = 'results'
    image_path = 'images'
    make_directories([results_path, image_path])

    json_file_name = mean_energy.get_json_file_name()
    is_simulation_done = check_if_file_has_data(json_file_name, sub_dir=results_path)
    ask_to_redo_simulation = False

    if is_simulation_done:
        ask_to_redo_simulation = ask_to_replace_file()

    if is_simulation_done is False or ask_to_redo_simulation:
        start_time = time()
        mean_energy.simulate_mean_energy()
        stop_time = time()
        mean_energy.logs = print_and_store(mean_energy.logs,
                                           message=f'Program took {round(stop_time - start_time, 3)} seconds.')
        mean_energy.save_data(sub_dir=results_path)
    else:
        mean_energy.load_data(sub_dir=results_path)
        print(mean_energy.logs)

    figure, axes = mean_energy.plot_mean_energy()

    image_name = os.path.join(image_path, mean_energy.file_name())
    if not check_if_file_exists(image_name):
        figure.savefig(image_name)
