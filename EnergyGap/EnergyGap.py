from HamiltonianClass import *
from time import time
import numpy as np
import matplotlib.pyplot as plt
import json


class EnergyGap(object):

    def __init__(self, J=0, L_max=0, delta_list=None, hamiltonian_reduced=False, divided_by_L=False, is_pbc=False):
        self.J = J
        self.L = L_max
        self.L_list = delete_odd_numbers(create_ascending_list(L_max))
        self.delta_list = delta_list
        self.hamiltonian_reduced = hamiltonian_reduced
        self.divided_by_L = divided_by_L
        self.pbc = is_pbc
        self.energy_gap_delta_dict = {}
        self.json_file = self.file_name(extension='.json')

    def simulate_energy_gap(self):
        for delta in self.delta_list:
            # energy_delta_range = np.zeros(len(self.L_list))
            energy_delta_range = []
            for index, L in enumerate(self.L_list):
                start_time_sim = time()
                quantum_state = QuantumState(L, self.J, delta, is_reduced=self.hamiltonian_reduced, is_pbc=self.pbc)
                energy_delta = quantum_state.get_energy_delta(0, 1)
                if self.divided_by_L:
                    energy_delta = energy_delta / L
                # energy_delta_range[index] = energy_delta
                energy_delta_range.append(energy_delta)
                stop_time_sim = time()
                print(f'Delta = {delta}, L = {L}, Energy delta = {round(energy_delta, 4)}, '
                      f'time elapsed: {round(stop_time_sim - start_time_sim, 3)} seconds')
            self.energy_gap_delta_dict[str(delta)] = energy_delta_range

    def plot_energy_gap(self):
        amount_of_deltas = len(self.delta_list)
        figure, axes = plt.subplots(1, amount_of_deltas)
        inverse_of_L = [round(1 / L, 4) for L in self.L_list]
        for delta in self.delta_list:
            energy_gaps = self.energy_gap_delta_dict[str(delta)]
            extrapolated_x, extrapolated_y, coefficients = extrapolate_data(inverse_of_L, energy_gaps)
            polynomial = polynomial_function_string(coefficients)
            index = self.delta_list.index(delta)
            axes[index].plot(extrapolated_x, extrapolated_y, label=polynomial)
            axes[index].scatter(inverse_of_L, energy_gaps, color='black')
            axes[index].set_title(f'$\Delta = {delta}$')
            axes[index].set(ylim=(0, max(energy_gaps) + 0.1))
            axes[index].grid()
            axes[index].legend(loc='upper left')
        axes[0].set(ylabel='Energy gap')
        figure.supxlabel('1 / L')
        figure.set_size_inches(7 * 1.92, 7 * 1.08)
        figure.suptitle(self.figure_title())
        return figure, axes

    def figure_title(self):
        title = 'Energy gap'
        if self.divided_by_L:
            title += ' / L'
        if self.hamiltonian_reduced:
            title += ', total spin = 0'
        else:
            title += ', full-size hamiltonian'
        if self.pbc:
            title += ', periodic boundary conditions'
        else:
            title += ', open boundary conditions'
        return title

    def file_name(self, extension='.png'):
        name = f'EnergyGap_L{self.L}D'
        for delta in self.delta_list:
            name += str(delta)
        if self.divided_by_L:
            name += '_divided_by_L'
        if self.hamiltonian_reduced:
            name += '_reduced'
        name += '_pbc' if self.pbc else '_obc'
        return name + extension

    def save_data(self):
        dict_with_data = {'deltas': self.delta_list,
                          'L list': self.L_list,
                          'J': self.J,
                          'L': self.L,
                          'energy gap': self.energy_gap_delta_dict}
        save_json_file(dict_with_data, self.json_file)

    def load_data(self):
        dict_with_data = read_json_file(self.json_file)
        self.delta_list = dict_with_data['deltas']
        self.L_list = dict_with_data['L list']
        self.J = dict_with_data['J']
        self.L = dict_with_data['L']
        self.energy_gap_delta_dict = dict_with_data['energy gap']

    def get_json_file_name(self):
        return self.json_file


if __name__ == '__main__':
    J = 1
    L = 14
    deltas = [0.5, 1.0, 2.0]
    is_reduced = False
    is_divided = False
    periodic_boundary_conditions = True

    energy_gap = EnergyGap(J=J, L_max=L, delta_list=deltas,
                           hamiltonian_reduced=is_reduced,
                           divided_by_L=is_divided,
                           is_pbc=periodic_boundary_conditions)

    json_file_name = energy_gap.get_json_file_name()
    is_simulation_done = check_if_file_has_data(json_file_name)
    ask_to_redo_simulation = False

    if is_simulation_done:
        ask_to_redo_simulation = ask_to_replace_file()

    if is_simulation_done is False or ask_to_redo_simulation:
        energy_gap.simulate_energy_gap()
        energy_gap.save_data()
    else:
        energy_gap.load_data()

    fig1, ax1 = energy_gap.plot_energy_gap()

    fig1.savefig(energy_gap.file_name())