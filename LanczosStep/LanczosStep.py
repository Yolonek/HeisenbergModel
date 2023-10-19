from HamiltonianClass import *
from CommonFunctions import print_and_store, make_directories
from time import time
import numpy as np
import matplotlib.pyplot as plt


class LanczosStep(object):
    def __init__(self, J=1, L=0, delta=0, hamiltonian_reduced=False,
                 is_pbc=False, lanczos_steps=0, random_param=0.5):
        self.J = J
        self.L = L
        self.delta = delta
        if hamiltonian_reduced:
            self.L = L if L % 2 == 0 else L - 1
        else:
            self.L = L
        self.hamiltonian_reduced = hamiltonian_reduced
        self.pbc = is_pbc
        self.lanczos_steps = lanczos_steps
        self.random_param = random_param
        self.x_index = None
        self.y_energies = None
        self.real_eigenvalues = None
        self.json_file = self.file_name(extension='.json')
        self.logs = ''

    def file_name(self, extension='.png'):
        name = f'LanczosStep_L{self.L}_LS{self.lanczos_steps}'
        if self.hamiltonian_reduced:
            name += '_reduced'
        name += '_pbc' if self.pbc else '_obc'
        return name + extension

    def get_json_file_name(self):
        return self.json_file

    def figure_title(self):
        title = (f'Lanczos eigenvalues convergence in function of '
                 f'${self.lanczos_steps}$ Lanczos steps, size $L={self.L}$')
        if self.hamiltonian_reduced:
            title += r', $\hat{S}^z_{tot} = 0$'
        else:
            title += ', full-size hamiltonian'
        if self.pbc:
            title += ', periodic boundary conditions'
        else:
            title += ', open boundary conditions'
        title += f', {len(self.real_eigenvalues)} real eigenvalues'
        return title

    def save_data(self, sub_dir=''):
        dict_with_data = {'J': self.J,
                          'delta': self.delta,
                          'L': self.L,
                          'eigenvalues': self.real_eigenvalues.tolist(),
                          'steps': self.lanczos_steps,
                          'x index': self.x_index.tolist(),
                          'lanczos eigenvalues': self.y_energies.tolist(),
                          'logs': self.logs}
        save_json_file(dict_with_data, self.json_file, sub_dir=sub_dir)

    def load_data(self, sub_dir=''):
        dict_with_data = read_json_file(self.json_file, sub_dir=sub_dir)
        self.J = dict_with_data['J']
        self.delta = dict_with_data['delta']
        self.L = dict_with_data['L']
        self.real_eigenvalues = np.array(dict_with_data['eigenvalues'])
        self.lanczos_steps = dict_with_data['steps']
        self.x_index = np.array(dict_with_data['x index'])
        self.y_energies = np.array(dict_with_data['lanczos eigenvalues'])
        self.logs = dict_with_data['logs']

    def simulate_lanczos_step(self, disable_print=False):
        self.logs = print_and_store(self.logs, disable_print=disable_print)
        quantum_state = QuantumState(self.L, self.J, self.delta,
                                     is_reduced=self.hamiltonian_reduced,
                                     is_pbc=self.pbc)
        quantum_state.set_random_state_vector(range_parameter=self.random_param)
        for i in range(1, self.lanczos_steps + 1):
            start_time_sim = time()
            quantum_state.lanczos_step()
            eigenvalues, eigenvectors = quantum_state.lanczos_matrix_eigenstates()
            if i == 1:
                self.x_index = np.array([i])
                self.y_energies = np.array([as_scalar(eigenvalues)])
            else:
                x_new_index = np.full(len(eigenvalues), i)
                y_new_energies = np.array(eigenvalues).flatten()
                self.x_index = np.concatenate((self.x_index, x_new_index), axis=0)
                self.y_energies = np.concatenate((self.y_energies, y_new_energies), axis=0)
            stop_time_sim = time()
            if i % 10 == 0:
                self.logs = print_and_store(self.logs,
                                            message=f'Lanczos steps completed: {i}, '
                                                    f'time taken: {round(stop_time_sim - start_time_sim, 4)} seconds',
                                            disable_print=disable_print)
        self.real_eigenvalues = np.array(quantum_state.get_all_eigenvalues()).flatten()

    def plot_lanczos_step(self):
        figure, axes = plt.subplots(1, 1, layout='constrained')
        axes.scatter(self.x_index, self.y_energies,
                     marker='+', color='black', label='Lanczos eigenvalues')
        real_eigenvalues_index = np.full(len(self.real_eigenvalues), self.lanczos_steps + 1)
        axes.scatter(real_eigenvalues_index, self.real_eigenvalues,
                     marker='x', color='blue', label='real eigenvalues')
        axes.set(xlabel='Lanczos step', ylabel='energies')
        axes.grid()
        axes.legend(loc='upper right')
        figure.suptitle(self.figure_title())
        figure.set_size_inches(8 * 2.56, 8 * 1.08)
        return figure, axes


if __name__ == '__main__':
    J = 1
    L = 10
    delta = 1
    lanczos_steps = 100

    periodic_boundary = False
    spin_zero = True

    lanczos = LanczosStep(J=J, L=L, delta=delta,
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
        lanczos.simulate_lanczos_step()
        stop_time = time()
        lanczos.logs = print_and_store(lanczos.logs,
                                       message=f'Program took {round(stop_time - start_time, 3)} seconds.')
        lanczos.save_data(sub_dir=results_path)
    else:
        lanczos.load_data(sub_dir=results_path)
        print(lanczos.logs)

    figure, axes = lanczos.plot_lanczos_step()

    image_name = os.path.join(image_path, lanczos.file_name())
    if not check_if_file_exists(image_name):
        figure.savefig(image_name)

