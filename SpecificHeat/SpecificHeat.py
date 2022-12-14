from HamiltonianClass import *
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

    def file_name(self, extension='.png'):
        name = f'SpecificHeat_L{self.L}_D{self.delta}'
        if self.hamiltonian_reduced:
            name += '_reduced'
        name += '_pbc' if self.pbc else '_obc'
        return name + extension

    def get_json_file_name(self):
        return self.json_file

    def figure_title(self):
        title = 'Graphs of Cv(T) and Cv(T) / L'
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
                          'L': self.L_list,
                          'reduced': self.hamiltonian_reduced,
                          'periodic': self.pbc,
                          'temperatures': self.temperatures.tolist(),
                          'specific heat': {}}
        for L, heat in self.specific_heat_dict.items():
            dict_with_data['specific heat'][L] = heat.tolist()
        save_json_file(dict_with_data, self.json_file)

    def load_data(self):
        dict_with_data = read_json_file(self.json_file)
        self.delta = dict_with_data['delta']
        self.L_list = dict_with_data['L']
        self.J = dict_with_data['J']
        self.hamiltonian_reduced = dict_with_data['reduced']
        self.pbc = dict_with_data['periodic']
        self.temperatures = np.array(dict_with_data['temperatures'])
        self.specific_heat_dict = {}
        for L, heat in dict_with_data['specific heat'].items():
            self.specific_heat_dict[L] = np.array(heat)

    def simulate_specific_heat(self):
        self.specific_heat_dict = {}
        for L in self.L_list:
            start_time_sim = time()
            quantum_state = QuantumState(L, self.J, self.delta, is_reduced=self.hamiltonian_reduced)
            quantum_state.calculate_specific_heat_range(self.temperatures)
            specific_heat = quantum_state.specific_heat
            self.specific_heat_dict[str(L)] = specific_heat
            stop_time_sim = time()
            print(f'L = {L}, time elapsed: {round(stop_time_sim - start_time_sim, 3)} seconds')

    def plot_specific_heat(self):
        figure, axis = plt.subplots(1, 2)
        for L in self.L_list:
            specific_heat = self.specific_heat_dict[str(L)]
            axis[0].plot(self.temperatures, specific_heat, label=f'L = {L}')
            axis[1].plot(self.temperatures, specific_heat / L, label=f'L = {L}')
        for index in range(2):
            axis[index].set_title(f'delta = {self.delta}')
            axis[index].legend(loc='upper right')
            axis[index].grid()
            axis[index].set(xlabel='T / J')
        axis[0].set(ylabel='Cv')
        figure.set_size_inches(6 * 2.56, 6 * 1.08)
        figure.suptitle(self.figure_title())
        return figure, axis


J = 1
L = 14
delta = 1
temperatures = linspace(0.1, 2, 100)
periodic_boundary = False
spin_zero = False

specific_heat = SpecificHeat(J=J, L=L, delta=delta,
                             hamiltonian_reduced=spin_zero,
                             is_pbc=periodic_boundary,
                             temperature_range=temperatures)

json_file_name = specific_heat.get_json_file_name()
is_simulation_done = check_if_file_has_data(json_file_name)
ask_to_redo_simulation = False

if is_simulation_done:
    ask_to_redo_simulation = ask_to_replace_file()

if is_simulation_done is False or ask_to_redo_simulation:
    specific_heat.simulate_specific_heat()
    specific_heat.save_data()
else:
    specific_heat.load_data()

fig1, ax1 = specific_heat.plot_specific_heat()

fig1.savefig(specific_heat.file_name())



