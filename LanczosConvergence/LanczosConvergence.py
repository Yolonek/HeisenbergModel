from HamiltonianClass import *
from CommonFunctions import print_and_store, make_directories
from time import time
import numpy as np
import matplotlib.pyplot as plt


class LanczosConvergence(object):
    def __init__(self, J=1, L_max=0, delta_list=None, hamiltonian_reduced=False,
                 is_pbc=False, lanczos_steps=0, random_param=0.5):
        self.J = J
        self.L = L_max
        self.deltas = delta_list
        if hamiltonian_reduced:
            self.L = L_max if L_max % 2 == 0 else L_max - 1
        else:
            self.L = L_max
        self.L_list = list(range(6, self.L + 1))
        if hamiltonian_reduced:
            self.L_list = delete_odd_numbers(self.L_list)
        self.hamiltonian_reduced = hamiltonian_reduced
        self.pbc = is_pbc
        self.lanczos_steps = lanczos_steps
        self.random_param = random_param
        self.lanczos_energy_dict = {}
        self.json_file = self.file_name(extension='.json')
        self.logs = ''

    def file_name(self, extension='.png'):
        name = f'LanczosConvergence_L{self.L}_LS{self.lanczos_steps}'
        if self.hamiltonian_reduced:
            name += '_reduced'
        name += '_pbc' if self.pbc else '_obc'
        return name + extension

    def get_json_file_name(self):
        return self.json_file

    def figure_title(self):
        title = f'Lanczos ground energy convergence in function of ' \
                f'${self.lanczos_steps}$ Lanczos steps, $L={self.L}$'
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
                          'L': self.L,
                          'L list': self.L_list,
                          'steps': self.lanczos_steps,
                          'ground states': self.lanczos_energy_dict,
                          'logs': self.logs}
        save_json_file(dict_with_data, self.json_file, sub_dir=sub_dir)

    def load_data(self, sub_dir=''):
        dict_with_data = read_json_file(self.json_file, sub_dir=sub_dir)
        self.J = dict_with_data['J']
        self.deltas = dict_with_data['deltas']
        self.L = dict_with_data['L']
        self.L_list = dict_with_data['L list']
        self.lanczos_steps = dict_with_data['steps']
        self.lanczos_energy_dict = dict_with_data['ground states']
        self.logs = dict_with_data['logs']

    def simulate_lanczos_convergence(self, disable_print=False):
        self.logs = print_and_store(self.logs, disable_print=disable_print)
        self.lanczos_energy_dict = {}
        for L in self.L_list:
            lanczos_energy_L = {}
            for delta in self.deltas:
                start_time_sim = time()
                lanczos_ground_energy = np.zeros(self.lanczos_steps)
                quantum_state = QuantumState(L, self.J, delta,
                                             is_reduced=self.hamiltonian_reduced,
                                             is_pbc=self.pbc)
                quantum_state.set_random_state_vector(self.random_param)
                real_ground_energy = quantum_state.get_nth_eigenvalue(0)
                for i in range(1, self.lanczos_steps + 1):
                    quantum_state.lanczos_step()
                    eigenvalues, eigenvectors = quantum_state.lanczos_matrix_eigenstates()
                    lanczos_ground_energy[i - 1] = eigenvalues[0]
                lanczos_energy_L[str(delta)] = (lanczos_ground_energy.tolist(), real_ground_energy)
                stop_time_sim = time()
                self.logs = print_and_store(self.logs,
                                            disable_print=disable_print,
                                            message=f'L = {L}, delta = {delta}, '
                                                    f'time taken: {round(stop_time_sim - start_time_sim, 4)} seconds')
            self.lanczos_energy_dict[str(L)] = lanczos_energy_L

    def plot_lanczos_convergence(self, set_up_figure=True):
        lanczos_range = list(range(1, self.lanczos_steps + 1))
        figure, axis = plt.subplots(1, 1, layout='constrained')
        for L, lanczos_delta_dict in self.lanczos_energy_dict.items():
            for delta, lanczos_ground_energy in lanczos_delta_dict.items():
                linestyle = 'solid' if delta == '0' else 'dashed'
                label = f'$L = {L}, \Delta = {delta}, \epsilon_0 = {round(lanczos_ground_energy[1], 4)}$'
                lanczos_converged = [energy / lanczos_ground_energy[1] for energy in lanczos_ground_energy[0]]
                axis.plot(lanczos_range, lanczos_converged, linestyle=linestyle, label=label)
        axis.set(xlabel='Lanczos step', ylabel=r'$\frac{\tilde{\epsilon}_0 - \epsilon_0}{\epsilon_0}$')
        axis.legend(loc='lower right')
        axis.grid()
        if set_up_figure:
            figure.suptitle(self.figure_title())
            figure.set_size_inches(8 * 1.92, 8 * 1.08)
        return figure, axis


if __name__ == '__main__':
    J = 1
    L = 14
    deltas = [0, 1]
    lanczos_steps = 50

    periodic_boundary = False
    spin_zero = True

    lanczos = LanczosConvergence(J=J, L_max=L, delta_list=deltas,
                                 is_pbc=periodic_boundary,
                                 hamiltonian_reduced=spin_zero,
                                 lanczos_steps=lanczos_steps)

    results_path = 'results'
    image_path = 'images'
    make_directories([results_path, image_path])

    json_file_name = lanczos.get_json_file_name()
    is_simulation_done = check_if_file_has_data(json_file_name, sub_dir=results_path)
    ask_to_redo_simulation = False

    if is_simulation_done:
        ask_to_redo_simulation = ask_to_replace_file()

    if is_simulation_done is False or ask_to_redo_simulation:
        start_time = time()
        lanczos.simulate_lanczos_convergence()
        stop_time = time()
        lanczos.logs = print_and_store(lanczos.logs,
                                       message=f'Program took {round(stop_time - start_time, 3)} seconds.')
        lanczos.save_data(sub_dir=results_path)
    else:
        lanczos.load_data(sub_dir=results_path)
        print(lanczos.logs)

    figure, axes = lanczos.plot_lanczos_convergence()

    image_name = os.path.join(image_path, lanczos.file_name())
    if not check_if_file_exists(image_name):
        figure.savefig(image_name)
