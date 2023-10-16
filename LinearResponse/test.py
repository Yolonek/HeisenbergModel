from HamiltonianClass import *
from CommonFunctions import *
import numpy as np
from time import time
import matplotlib.pyplot as plt


def dirac_delta_approximation(x, epsilon=1e-4):
    return np.exp(-x ** 2 / (2 * epsilon ** 2)) / (epsilon * np.sqrt(2 * np.pi))

if __name__ == '__main__':
    J = 1
    delta = 1
    L = 5
    temperature_range = [0, 1, float('inf')]
    temp = float('inf')

    periodic_boundary = True
    spin_zero = True
    omega_bin = 0.1
    k = 1
    om = 0

    quantum_state = QuantumState(L, J, delta, is_pbc=periodic_boundary,
                                 is_reduced=spin_zero)
    omega_range = quantum_state.generate_linspace_of_omega(omega_bin)
    # quantum_state.set_sq_operator(wave_vector(L, k))
    grid = quantum_state.create_eigenvalues_histogram_map(difference_function, omega_bin)
    quantum_state.print_eigenvalues()
    print(grid)

    # def my_function(x, y):
    #     return x + y
    #
    # x = np.array([1, 2, 3])
    # y = np.array([4, 5, 6])
    #
    # X, Y = np.meshgrid(x, x)
    #
    # result_matrix = my_function(X, Y)
    #
    # print(result_matrix)


