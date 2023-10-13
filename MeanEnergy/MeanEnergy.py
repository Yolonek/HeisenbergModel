from HamiltonianClass import *
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

    def file_name(self, extension='.png'):
        name = f'MeanEnergy_L{self.L}D'
        for delta in self.deltas:
            name += str(delta)
        if self.divided_by_L:
            name += '_divided_by_L'
        if self.hamiltonian_reduced:
            name += '_reduced'
        name += '_pbc' if self.pbc else '_obc'
        return name + extension

    def get_json_file_name(self):
        return self.json_file

    def figure_title(self):
        title = 'Graph of mean energy for each delta'
        if self.divided_by_L:
            title += ', divided by L'
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
                          'deltas': self.deltas,
                          'L list': self.L_list,
                          'L': self.L,
                          'reduced': self.hamiltonian_reduced,
                          'periodic': self.pbc,
                          'divided': self.divided_by_L,
                          'temperatures': self.temperatures.tolist(),
                          'mean energy': {}}
        for delta in self.mean_energy_dict:
            dict_with_data['mean energy'][delta] = {}
            for L, energy in self.mean_energy_dict[delta].items():
                dict_with_data['mean energy'][delta][L] = energy.tolist()
        save_json_file(dict_with_data, self.json_file)

    def load_data(self):
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

    def simulate_mean_energy(self):
        self.mean_energy_dict = {}
        for delta in self.deltas:
            self.mean_energy_dict[str(delta)] = {}
            for L in self.L_list:
                start_time_sim = time()
                quantum_state = QuantumState(L, self.J, delta, is_reduced=self.hamiltonian_reduced)
                quantum_state.calculate_mean_energy_range(self.temperatures)
                mean_energy = quantum_state.mean_energy
                if self.divided_by_L:
                    mean_energy = mean_energy / L
                self.mean_energy_dict[str(delta)][str(L)] = mean_energy
                stop_time_sim = time()
                print(f'Delta = {delta}, L = {L}, time elapsed: {round(stop_time_sim - start_time_sim, 3)} seconds')

    def plot_mean_energy(self):
        amount_of_deltas = len(self.deltas)
        figure, axis = plt.subplots(1, amount_of_deltas)
        for delta, index in zip(self.deltas, range(amount_of_deltas)):
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

    json_file_name = mean_energy.get_json_file_name()
    is_simulation_done = check_if_file_has_data(json_file_name)
    ask_to_redo_simulation = False

    if is_simulation_done:
        ask_to_redo_simulation = ask_to_replace_file()

    if is_simulation_done is False or ask_to_redo_simulation:
        mean_energy.simulate_mean_energy()
        mean_energy.save_data()
    else:
        mean_energy.load_data()

    fig1, ax1 = mean_energy.plot_mean_energy()

    fig1.savefig(mean_energy.file_name())