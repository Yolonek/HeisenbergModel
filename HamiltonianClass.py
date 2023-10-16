from CommonFunctions import *
from OperatorFunctions import *
from pyarma import mat, eig_sym, norm, fill
import numpy as np
from math import exp, pi, cos
from cmath import exp as c_exp
from time import time
from random import uniform
from matplotlib import pyplot as plt


class Hamiltonian(object):

    def __init__(self, L, J, delta, is_pbc=False):
        self.L = L
        self.J = J
        self.delta = delta
        self.size = 2 ** L
        self.matrix = mat(2 ** L, 2 ** L)
        self.basis = generate_binary_strings(L)
        self.pbc = is_pbc
        self.reduced = False
        self.eigenvalues = None
        self.eigenvectors = None
        self.mean_energy = None
        self.specific_heat = None

    def truncate_basis_to_spin_zero(self):
        if self.L % 2 == 0:
            new_basis = []
            for basis in self.basis:
                if s_z_total(basis) == 0:
                    new_basis.append(basis)
            self.basis = new_basis
            self.size = len(new_basis)
            self.matrix = mat(self.size, self.size)
            self.reduced = True
        else:
            print("[ERROR] Can't truncate odd-number state")

    def print_hamiltonian_data(self, return_msg=False):
        msg = f'L = {self.L}, J = {self.J}, delta = {self.delta}, ' \
              f'hamiltonian size: {self.size} x {self.size}'
        print(msg)
        if return_msg:
            return msg

    def print_basis(self):
        for basis_state in self.basis:
            print(f'|{basis_state}>')

    def print_matrix(self):
        print(f'Matrix size: {self.size} x {self.size}')
        if self.reduced:
            print('Basis reduced to spin zero:')
        else:
            print('Basis:')
        print(end=(9 - self.L) * ' ')
        for i in range(self.size):
            print(self.basis[i], end=(9 - self.L) * ' ')
        print('\n')
        self.matrix.print()

    def plot_data(self, matrix=None, axes=None):
        if axes:
            axes.axis('off')

            matrix = np.round(np.array(matrix), 4) if matrix else np.round(np.array(self.matrix), 4)

            table = axes.table(cellText=matrix, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)

            cell_height = 1 / matrix.shape[0]
            cell_width = 1 / matrix.shape[1]

            for cell in table._cells.values():
                cell.set_edgecolor('none')
                cell.set_linewidth(0)
                cell.set_height(cell_height)
                cell.set_width(cell_width)

    def hamiltonian_p_element(self, basis_index):
        matrix_sum = 0
        for i in range(len(self.basis[basis_index]) - 1):
            matrix_sum += s_z(i, self.basis[basis_index]) * s_z(i + 1, self.basis[basis_index])
        if self.pbc:
            matrix_sum += s_z(-1, self.basis[basis_index]) * s_z(0, self.basis[basis_index])
        return matrix_sum * self.J * self.delta

    def prepare_hamiltonian_p(self):
        hamiltonian_p = mat(self.size, self.size)
        for i in range(len(self.basis)):
            hamiltonian_p[i, i] += self.hamiltonian_p_element(i)
        self.matrix += hamiltonian_p

    def prepare_hamiltonian_k(self):
        hamiltonian_k = mat(self.size, self.size)
        for basis_el in self.basis:
            combinations = calculate_interaction(basis_el, self.pbc)
            j = self.basis.index(basis_el)
            for result in combinations:
                if result != '':
                    i = self.basis.index(result)
                    hamiltonian_k[i, j] += self.J / 2
        self.matrix += hamiltonian_k

    def heisenberg_hamiltonian(self):
        self.reset_hamiltonian()
        self.prepare_hamiltonian_p()
        self.prepare_hamiltonian_k()

    def reset_hamiltonian(self):
        self.matrix.zeros()

    def eigenstates(self, add_perturbation=None):
        if self.matrix.is_zero():
            print('[ERROR] Hamiltonian matrix is empty')
        else:
            self.eigenvalues = mat()
            self.eigenvectors = mat()
            if add_perturbation is None:
                eig_sym(self.eigenvalues, self.eigenvectors, self.matrix)
            else:
                eig_sym(self.eigenvalues, self.eigenvectors, self.matrix + add_perturbation)

    def print_eigenstates(self, only_eigenvalues=False):
        print('Eigenvalues:')
        self.eigenvalues.print()
        if not only_eigenvalues:
            print('Eigenvectors:')
            self.eigenvectors.print()

    def print_eigenvalues(self):
        self.eigenvalues.print()

    def print_eigenvectors(self):
        self.eigenvectors.print()

    def get_nth_eigenvector(self, n_level):
        return self.eigenvectors[:, n_level]

    def get_all_eigenvectors(self):
        return self.eigenvectors

    def get_nth_eigenvalue(self, n_level):
        return self.eigenvalues[n_level]

    def get_all_eigenvalues(self):
        return self.eigenvalues

    def get_hamiltonian(self):
        return self.matrix

    def get_energy_delta(self, level_1, level_2):
        if self.eigenvalues is not None:
            return round(self.eigenvalues[level_2] - self.eigenvalues[level_1], 8)
        else:
            return 0

    def calculate_ensemble(self, temperature):
        return np.sum(1 / np.exp(self.eigenvalues / temperature))

    def calculate_mean_energy_power(self, temperature, power):
        ensemble = self.calculate_ensemble(temperature)
        exponent = 1 / np.exp(self.eigenvalues / temperature)
        if power != 1:
            sum_ = np.sum(np.power(self.eigenvalues, power) * exponent / ensemble)
        else:
            sum_ = np.sum(self.eigenvalues * exponent / ensemble)
        return sum_

    def calculate_mean_energy_range(self, temperature_range):
        temp_length = len(temperature_range)
        mean_energy_range = np.zeros(temp_length)
        for temperature, i in zip(temperature_range, range(temp_length)):
            mean_energy = self.calculate_mean_energy_power(temperature, 1)
            mean_energy_range[i] = mean_energy
        self.mean_energy = mean_energy_range

    def calculate_specific_heat(self, temperature, normalize=False):
        mean_energy = self.calculate_mean_energy_power(temperature, 1)
        mean_energy_sq = self.calculate_mean_energy_power(temperature, 2)
        variance = mean_energy_sq - (mean_energy ** 2)
        specific_heat = variance / (temperature ** 2)
        return specific_heat / self.L if normalize else specific_heat

    def calculate_specific_heat_range(self, temperature_range):
        specific_heat_range = np.zeros(len(temperature_range))
        for i, temperature in enumerate(temperature_range):
            specific_heat = self.calculate_specific_heat(temperature)
            specific_heat_range[i] = specific_heat
        self.specific_heat = specific_heat_range


class QuantumState(Hamiltonian):

    def __init__(self, L, J, delta, is_pbc=False, is_reduced=False, vector_state=None):
        super().__init__(L, J, delta, is_pbc)
        if is_reduced:
            self.truncate_basis_to_spin_zero()
        self.expected_value = None
        self.variance = None
        self.heisenberg_hamiltonian()
        self.eigenstates()
        if vector_state is None:
            self.state_vector = self.get_nth_eigenvector(0)
        else:
            self.state_vector = vector_state
        self.operator = None
        self.lanczos_matrix = None
        self.lanczos_steps = 0
        self.lanczos_vector = None
        self.lanczos_previous_vector = None

    def calculate_expected_value(self, vector=None, assign=True):
        if vector is None:
            vector = self.state_vector
        left_element = trans(vector)
        expected_value = left_element * (self.matrix * vector)
        if assign:
            self.expected_value = round(as_scalar(expected_value), 8)
        else:
            return round(as_scalar(expected_value), 8)

    def get_expected_value(self):
        return self.expected_value

    def calculate_variance(self):
        left_element = trans(self.state_vector)
        square_of_matrix = self.matrix * self.matrix
        expected_of_square = left_element * (square_of_matrix * self.state_vector)
        if self.expected_value is None:
            self.calculate_expected_value()
        square_of_expected = self.get_expected_value() ** 2
        self.variance = round(as_scalar(expected_of_square - square_of_expected), 6)

    def get_variance(self):
        return self.variance

    def set_vector_from_eigenstate(self, eigenvector_number):
        self.state_vector = self.get_nth_eigenvector(eigenvector_number)

    def set_state_vector(self, pyarma_vec):
        self.state_vector = pyarma_vec

    def print_state_vector(self):
        vector_to_print = f''
        for index, basis_element in enumerate(self.basis):
            vector_value = round(self.state_vector[index], 4)
            if vector_value != 0:
                if len(vector_to_print) == 0:
                    vector_to_print += '- ' if vector_value < 0 else ''
                    vector_to_print += f'{abs(vector_value)}|{basis_element}>'
                else:
                    vector_to_print += ' + ' if vector_value > 0 else ' - '
                    vector_to_print += f'{abs(vector_value)}|{basis_element}>'
        print(vector_to_print)

    def print_all_data(self, with_matrix=False, with_state_vector=False):
        self.print_hamiltonian_data()
        if with_state_vector:
            print('Current state vector:')
            self.print_state_vector()
        if self.expected_value is not None:
            print(f'Expected value: {self.expected_value}')
        if self.variance is not None:
            print(f'Variance: {self.variance}')
        if with_matrix:
            self.print_matrix()

    def set_basis_element_to_state_vector(self, basis_element):
        if basis_element in self.basis:
            vector_length = len(self.basis)
            basis_index = self.basis.index(basis_element)
            basis_vector = mat()
            basis_vector.zeros(vector_length, 1)
            basis_vector[basis_index] = 1
            self.state_vector = basis_vector
        else:
            print('[ERROR] Given state is not in current basis')

    def get_state_vector(self):
        return self.state_vector

    def find_temperature_for_state_energy(self, temperature_range):
        start_time = time()
        self.calculate_expected_value()
        self.calculate_mean_energy_range(temperature_range)
        energy_of_state = round(self.expected_value / self.L, 4)
        found_index = find_nearest_value(self.mean_energy / self.L, energy_of_state)
        found_temperature = round(temperature_range[found_index].item(), 4)
        stop_time = time()
        print(f'Temperature corresponding to energy {energy_of_state} is: {found_temperature},'
              f' time elapsed: {round(stop_time - start_time, 3)} seconds')
        return found_temperature, energy_of_state

    def coefficient_c_of_t(self, time_value, eigenstate_number):
        nth_eigenvalue = self.get_nth_eigenvalue(eigenstate_number)
        nth_eigenvector = self.get_nth_eigenvector(eigenstate_number)
        exponent_index = complex(0, nth_eigenvalue * time_value)
        exponent = c_exp(exponent_index)
        product = trans(nth_eigenvector) * self.state_vector
        return exponent * as_scalar(product)

    def coefficient_c_of_t_for_all_eigenvalues(self, time_value):
        product_vector = self.get_state_vector().t() * self.get_all_eigenvectors()
        exponent_vector = cx_mat(mat(len(self.eigenvalues), 1, fill.zeros),
                                 self.eigenvalues * time_value)
        exponent_vector.transform(c_exp)
        return exponent_vector @ product_vector.t()

    def quantum_state_of_t(self, time_value):
        coefficient_vector = np.array(self.coefficient_c_of_t_for_all_eigenvalues(time_value)).flatten()
        eigenvector_matrix = np.array(self.get_all_eigenvectors())
        state_vector_matrix = eigenvector_matrix * coefficient_vector
        return cx_mat(state_vector_matrix.sum(axis=1)[:, None])

    def quantum_state_of_t_deprecated(self, time_value):
        number_of_states = len(self.eigenvalues)
        quantum_state = cx_mat()
        quantum_state.zeros(number_of_states, 1)
        for state_number in range(number_of_states):
            coefficient = self.coefficient_c_of_t(time_value, state_number)
            eigenvector = self.get_nth_eigenvector(state_number)
            quantum_state += coefficient * cx_mat(eigenvector)
        return quantum_state

    def set_spin_operator(self, spin_number, assign=True):
        spin_operator = mat()
        spin_operator.zeros(self.size, self.size)
        for index, basis_state in enumerate(self.basis):
            spin_operator[index, index] = s_z(spin_number, basis_state)
        if assign:
            self.operator = spin_operator
        else:
            return spin_operator

    def operator_time_evolution(self, time_value):
        quantum_state = self.quantum_state_of_t(time_value)
        operator_mean_value = calculate_expected_value(cx_mat(self.operator), quantum_state)
        return operator_mean_value.real

    def operator_time_evolution_vectorized(self):
        return np.vectorize(self.operator_time_evolution)

    def calculate_operator_time_evolution(self, time_range):
        operator_mean_value_numpy = np.zeros(len(time_range))
        for index, dt in enumerate(time_range):
            quantum_state = self.quantum_state_of_t(dt)
            operator_mean_value = calculate_expected_value(self.operator, quantum_state)
            operator_mean_value_numpy[index] = operator_mean_value.real
        return operator_mean_value_numpy

    def set_sq_operator(self, q, assign=True):
        sq_operator = cx_mat()
        sq_operator.zeros(self.size, self.size)
        for i in range(self.L):
            spin_operator = self.set_spin_operator(i, assign=False)
            exponent_index = complex(0, q * i)
            exponent = c_exp(exponent_index)
            sq_operator += exponent * cx_mat(spin_operator)
        if assign:
            self.operator = sq_operator
        else:
            return sq_operator

    def calculate_linear_response(self, omega_range, omega_bin, temperature):
        dirac_delta_func = np.vectorize(dirac_delta_function)
        linear_response = np.zeros(len(omega_range))
        ensemble = self.calculate_ensemble(temperature) if temperature != 0 else 1
        for n in range(self.size):
            n_value = self.get_nth_eigenvalue(n)
            n_state = self.get_nth_eigenvector(n)
            exponent = 1 / exp(n_value / temperature) if temperature != 0 else 1
            for m in range(self.size):
                m_value = self.get_nth_eigenvalue(m)
                m_state = self.get_nth_eigenvector(m)
                omega = m_value - n_value
                dirac_delta = dirac_delta_func(omega_range, omega, omega_bin)
                matrix_element = calculate_matrix_element(n_state, self.operator, m_state)
                matrix_element_square = square_of_complex_modulus(matrix_element)
                linear_response += dirac_delta * (exponent / ensemble) * matrix_element_square
            if temperature == 0 and n == 0:
                break
        return linear_response

    def generate_linspace_of_omega(self, step, positive_domain=False, boundary=None):
        if boundary is None:
            ground_energy = self.get_nth_eigenvalue(0)
            last_eigenvalue = self.size - 1
            omega_max = self.get_nth_eigenvalue(last_eigenvalue) - ground_energy
            if positive_domain:
                omega_space = np.arange(0, omega_max, step)
            else:
                omega_space = np.arange(-omega_max, omega_max, step)
            return omega_space
        else:
            return np.arange(boundary[0], boundary[1], step)

    def get_wave_vector_perturbation(self, h, wave_vec):
        perturbation = mat()
        perturbation.zeros(self.size, self.size)
        for spin in range(self.L):
            spin_operator = self.set_spin_operator(spin, assign=False)
            perturbation += cos(wave_vec * spin) * spin_operator
        return h * perturbation

    def spin_evolution_range(self, time_range):
        time_evolution_function = np.vectorize(self.operator_time_evolution)
        time_evolution_dict = {}
        for spin in range(self.L):
            self.set_spin_operator(spin)
            time_evolution = time_evolution_function(time_range)
            time_evolution_dict[spin] = time_evolution
        return time_evolution_dict

    @staticmethod
    def sum_over_time(omega, time_range_im, operator_evolution):
        return np.sum(np.exp(omega * time_range_im) * operator_evolution)

    def calculate_linear_response_fft(self, omega_range, time_range_im, time_evo_dict, wave_vector):
        array_length = len(omega_range)
        linear_respones_q = np.zeros(array_length)
        for omega, index in zip(omega_range, range(array_length)):
            linear_response = 0
            for spin, operator_evolution in time_evo_dict.items():
                exponent_index = complex(0, wave_vector * spin)
                linear_response += c_exp(exponent_index) * self.sum_over_time(omega, time_range_im, operator_evolution)
            linear_respones_q[index] = abs(linear_response) ** 2
        return linear_respones_q

    def set_random_state_vector(self, range_parameter, assign=True):
        new_vector = mat(self.size, 1)
        new_vector.imbue(lambda: uniform(0, 1) - range_parameter)
        new_vector = new_vector / norm(new_vector)
        if assign:
            self.state_vector = new_vector
        else:
            return new_vector

    def lanczos_step(self):
        if self.lanczos_matrix is None and self.lanczos_steps == 0:
            if self.expected_value is None:
                self.calculate_expected_value()
            self.lanczos_matrix = mat([self.expected_value])
            self.lanczos_vector = self.state_vector
        else:
            a_previous = self.lanczos_matrix[self.lanczos_steps - 1, self.lanczos_steps - 1]
            self.lanczos_matrix.resize(self.lanczos_steps + 1, self.lanczos_steps + 1)
            tmp = self.matrix * self.lanczos_vector
            if self.lanczos_steps == 1:
                tmp2 = tmp - a_previous * self.lanczos_vector
            else:
                b_previous = self.lanczos_matrix[self.lanczos_steps - 1, self.lanczos_steps - 2]
                tmp2 = tmp - a_previous * self.lanczos_vector - b_previous * self.lanczos_previous_vector
            bi = norm(tmp2)
            self.lanczos_matrix[self.lanczos_steps, self.lanczos_steps - 1] = bi
            self.lanczos_matrix[self.lanczos_steps - 1, self.lanczos_steps] = bi
            self.lanczos_previous_vector = self.lanczos_vector
            self.lanczos_vector = tmp2 / bi
            ai = self.calculate_expected_value(vector=self.lanczos_vector, assign=False)
            self.lanczos_matrix[self.lanczos_steps, self.lanczos_steps] = ai
        self.lanczos_steps += 1

    def lanczos_reset(self):
        self.lanczos_steps = 0
        self.lanczos_matrix = None
        self.lanczos_vector = None
        self.lanczos_previous_vector = None

    def print_lanczos_data(self, with_matrix=False, with_state_vector=False):
        print(f'Lanczos steps done: {self.lanczos_steps}')
        if with_matrix and self.lanczos_matrix is not None:
            print(f'Lanczos matrix size {self.lanczos_matrix.n_cols}x{self.lanczos_matrix.n_rows}:')
            self.lanczos_matrix.print()
        if with_state_vector and self.lanczos_vector is not None:
            print('Last Lanczos vector:')
            self.lanczos_vector.print()

    def do_n_lanczos_steps(self, steps):
        for n in range(steps):
            self.lanczos_step()

    def lanczos_matrix_eigenstates(self):
        eigenvalues, eigenvectors = mat(), mat()
        eig_sym(eigenvalues, eigenvectors, self.lanczos_matrix)
        return eigenvalues, eigenvectors

    def get_lanczos_matrix(self):
        return self.lanczos_matrix


if __name__ == '__main__':
    J = 1
    L = 4
    delta = 1
    periodic_boundary = False
    spin_zero = True

    state_1 = QuantumState(L, J, delta, is_pbc=periodic_boundary)
    state_1.calculate_expected_value()
    state_1.calculate_variance()
    state_1.print_all_data(with_matrix=True, with_state_vector=True)

    print('==========================================================================')

    state_2 = QuantumState(L, J, delta, is_pbc=periodic_boundary, is_reduced=spin_zero)
    state_2.calculate_expected_value()
    state_2.calculate_variance()
    state_2.print_all_data(with_matrix=True, with_state_vector=True)
