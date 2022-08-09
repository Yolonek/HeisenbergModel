from HamiltonianClass import *
from time import time
import numpy as np
import matplotlib.pyplot as plt


class TimeEvolution(object):

    def __init__(self, J=1, L=0, deltas=None, time_range=None, hamiltonian_reduced=False, is_pbc=False, initial_state=None):
        self.J = J
        self.deltas = deltas
        if hamiltonian_reduced:
            self.L = L if L % 2 == 0 else L - 1
        else:
            self.L = L
        if initial_state is None:
            self.initial_state = create_initial_state(self.L)
        else:
            self.initial_state = initial_state
        self.initial_energy = {}
        self.initial_temperature = {}
        self.hamiltonian_reduced = hamiltonian_reduced
        self.pbc = is_pbc
        self.time_evolution_dict = {}
        self.json_file = self.file_name(extension='.json')
        self.time_range = time_range

    def file_name(self, extension='.png'):
        name = f'TimeEvolution_L{self.L}'
        if self.hamiltonian_reduced:
            name += '_ham_reduced'
        if self.pbc:
            name += '_pbc'
        return name + extension

    def get_json_file_name(self):
        return self.json_file

    def figure_title(self):
        title = f'Magnetization in function of time for every spin with initial state |{self.initial_state}>'
        if self.hamiltonian_reduced:
            title += ', total spin = 0'
        else:
            title += ', full-size hamiltonian'
        return title

    def save_data(self):
        dict_with_data = {'J': self.J,
                          'deltas': self.deltas,
                          'L': self.L,
                          'reduced': self.hamiltonian_reduced,
                          'periodic': self.pbc,
                          'initial state': self.initial_state,
                          'initial energy': self.initial_energy,
                          'initial temperature': self.initial_temperature,
                          'time': self.time_range.tolist(),
                          'time evolution': {}}
        for delta, evo in self.time_evolution_dict.items():
            dict_with_data['time evolution'][delta] = evo.tolist()
        save_json_file(dict_with_data, self.json_file)

    def load_data(self):
        dict_with_data = read_json_file(self.json_file)
        self.J = dict_with_data['J']
        self.deltas = dict_with_data['deltas']
        self.L = dict_with_data['L']
        self.hamiltonian_reduced = dict_with_data['reduced']
        self.pbc = dict_with_data['periodic']
        self.initial_state = dict_with_data['initial state']
        self.initial_energy = dict_with_data['initial energy']
        self.initial_temperature = dict_with_data['initial temperature']
        self.time_range = np.array(dict_with_data['time'])
        self.time_evolution_dict = {}
        for delta, evo in dict_with_data['time evolution'].items():
            self.time_evolution_dict[delta] = np.array(evo)

    def simulate_time_evolution(self):
        self.time_evolution_dict = {}
        temperature_range = linspace(-4, -0.1, 200)
        for delta in self.deltas:
            quantum_state = QuantumState(self.L, self.J, delta,
                                         is_pbc=self.pbc,
                                         is_reduced=self.hamiltonian_reduced)
            quantum_state.set_basis_element_to_state_vector(self.initial_state)
            initial_temperature, initial_energy = quantum_state.find_temperature_for_state_energy(temperature_range)
            self.initial_energy[str(delta)] = initial_energy
            self.initial_temperature[str(delta)] = initial_temperature
            time_evolution_function = np.vectorize(quantum_state.operator_time_evolution)
            time_evolution_grid = None
            for spin_number in range(self.L):
                start_time_sim = time()
                quantum_state.set_spin_operator(spin_number)
                if spin_number == 0:
                    time_evolution_grid = time_evolution_function(self.time_range)[:, None]
                else:
                    time_evolution = time_evolution_function(self.time_range)[:, None]
                    time_evolution_grid = np.concatenate((time_evolution_grid, time_evolution), axis=1)
                stop_time_sim = time()
                print(f'Delta = {delta}, spin = {spin_number + 1}, '
                      f'time elapsed: {round(stop_time_sim - start_time_sim, 3)} seconds')
            self.time_evolution_dict[str(delta)] = time_evolution_grid

    def plot_time_evolution(self):
        spin_range = np.arange(1, self.L + 1, 1)
        amount_of_deltas = len(self.deltas)
        figure, axis = plt.subplots(1, amount_of_deltas)
        for delta, index in zip(self.deltas, range(amount_of_deltas)):
            initial_energy = self.initial_energy[str(delta)]
            initial_temperature = self.initial_temperature[str(delta)]
            time_evolution = self.time_evolution_dict[str(delta)]
            graph_title = f'delta = {delta},\ninitial energy = {initial_energy},' \
                          f'\ninitial temperature = {initial_temperature}'
            axis[index].set_title(graph_title)
            spin_max, spin_min = 0.5, -0.5
            grid = axis[index].pcolormesh(spin_range, self.time_range, time_evolution, cmap='RdBu',
                                          vmin=spin_min, vmax=spin_max)
            if axis[index] == axis[-1]:
                figure.colorbar(grid, ax=axis[index])
            axis[int(amount_of_deltas / 2)].set(xlabel='ith state')
        axis[0].set(ylabel='Time')
        figure.suptitle(self.figure_title())
        figure.set_size_inches(8 * 1.92, 8 * 1.08)
        plt.tight_layout()
        return figure, axis


J = 1
L = 16
deltas = [0, 1, 2]
periodic_boundary = False
spin_zero = True
times = linspace(0, 20, 200)
init_state = None

start_time = time()

time_evolution = TimeEvolution(J=J, L=L, deltas=deltas,
                               hamiltonian_reduced=spin_zero,
                               is_pbc=periodic_boundary,
                               time_range=times,
                               initial_state=init_state)

json_file_name = time_evolution.get_json_file_name()
is_simulation_done = check_if_file_has_data(json_file_name)
ask_to_redo_simulation = False

if is_simulation_done:
    ask_to_redo_simulation = ask_to_replace_file()

start_time = time()
if is_simulation_done is False or ask_to_redo_simulation:
    time_evolution.simulate_time_evolution()
    time_evolution.save_data()
else:
    time_evolution.load_data()

stop_time = time()
print(f'Program took {round(stop_time - start_time, 3)} seconds.')

fig1, ax1 = time_evolution.plot_time_evolution()

fig1.savefig(time_evolution.file_name())
