from HamiltonianClass import *
from CommonFunctions import print_and_store
from time import time
import matplotlib.pyplot as plt


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
        self.logs = ''

    def simulate_energy_gap(self):
        self.logs = print_and_store(self.logs)
        for delta in self.delta_list:
            energy_delta_range = []
            for index, L in enumerate(self.L_list):
                start_time_sim = time()
                quantum_state = QuantumState(L, self.J, delta, is_reduced=self.hamiltonian_reduced, is_pbc=self.pbc)
                self.logs = quantum_state.print_hamiltonian_data(return_msg=True)
                energy_delta = quantum_state.get_energy_delta(0, 1)
                if self.divided_by_L:
                    energy_delta = energy_delta / L
                energy_delta_range.append(energy_delta)
                stop_time_sim = time()
                self.logs = print_and_store(self.logs,
                                            message=f'Energy delta = {round(energy_delta, 4)}'
                                            f', time elapsed: {round(stop_time_sim - start_time_sim, 3)} seconds')
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
        title = r'$\epsilon_n(L)$'
        if self.divided_by_L:
            title += '$ / L$'
        if self.hamiltonian_reduced:
            title += r', $\hat{S}_{tot}^z = 0'
        else:
            title += ', full-size hamiltonian'
        if self.pbc:
            title += ', periodic boundary conditions'
        else:
            title += ', open boundary conditions'
        return title

    def file_name(self, extension='.png'):
        name = f'EnergyGap_L{self.L}D-'
        for delta in self.delta_list:
            name += f'{delta}-'
        name += 'pbc' if self.pbc else 'obc'
        if self.divided_by_L:
            name += '_divided_by_L'
        if self.hamiltonian_reduced:
            name += '_reduced'
        return name + extension

    def save_data(self, sub_dir=''):
        dict_with_data = {'deltas': self.delta_list,
                          'L list': self.L_list,
                          'J': self.J,
                          'L': self.L,
                          'energy gap': self.energy_gap_delta_dict,
                          'logs': self.logs}
        save_json_file(dict_with_data, self.json_file, sub_dir=sub_dir)

    def load_data(self, sub_dir=''):
        dict_with_data = read_json_file(self.json_file, sub_dir=sub_dir)
        self.delta_list = dict_with_data['deltas']
        self.L_list = dict_with_data['L list']
        self.J = dict_with_data['J']
        self.L = dict_with_data['L']
        self.energy_gap_delta_dict = dict_with_data['energy gap']
        self.logs = dict_with_data['logs']

    def get_json_file_name(self):
        return self.json_file


if __name__ == '__main__':
    J = 1
    L = 12
    deltas = [0.5, 1.0, 2.0]
    is_reduced = False
    is_divided = False
    periodic_boundary = True

    energy_gap = EnergyGap(J=J, L_max=L, delta_list=deltas,
                           hamiltonian_reduced=is_reduced,
                           divided_by_L=is_divided,
                           is_pbc=periodic_boundary)

    results_path = 'results'
    image_path = 'images'
    make_directories([results_path, image_path])

    json_file_name = energy_gap.get_json_file_name()
    is_simulation_done = check_if_file_has_data(json_file_name, sub_dir=results_path)
    ask_to_redo_simulation = False

    if is_simulation_done:
        ask_to_redo_simulation = ask_to_replace_file()

    if is_simulation_done is False or ask_to_redo_simulation:
        start_time = time()
        energy_gap.simulate_energy_gap()
        stop_time = time()
        energy_gap.logs = print_and_store(energy_gap.logs,
                                          message=f'Program took {round(stop_time - start_time, 3)} seconds.')
        energy_gap.save_data(sub_dir=results_path)
    else:
        energy_gap.load_data(sub_dir=results_path)
        print(energy_gap.logs)

    figure, axes = energy_gap.plot_energy_gap()

    image_name = os.path.join(image_path, energy_gap.file_name())
    if not check_if_file_exists(image_name):
        figure.savefig(image_name)
