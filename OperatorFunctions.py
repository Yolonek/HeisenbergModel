from pyarma import trans, as_scalar, cx_mat, mat, kron
from math import pi


def splus_sminus(spin_number, spins):
    new_state = spins
    if spin_number < len(new_state) - 1 and new_state != '':
        if new_state[spin_number:spin_number + 2] == '01':
            temp_state = list(new_state)
            temp_state[spin_number] = '1'
            temp_state[spin_number + 1] = '0'
            new_state = ''.join(temp_state)
        else:
            new_state = ''
    return new_state


def splus_sminus_boundary(spins):
    new_state = spins
    if new_state != '':
        if new_state[-1] == '0' and new_state[0] == '1':
            temp_state = list(new_state)
            temp_state[-1] = '1'
            temp_state[0] = '0'
            new_state = ''.join(temp_state)
        else:
            new_state = ''
    return new_state


def sminus_splus(spin_number, spins):
    new_state = spins
    if spin_number < len(new_state) - 1 and new_state != '':
        if new_state[spin_number:spin_number + 2] == '10':
            temp_state = list(new_state)
            temp_state[spin_number] = '0'
            temp_state[spin_number + 1] = '1'
            new_state = ''.join(temp_state)
        else:
            new_state = ''
    return new_state


def sminus_splus_boundary(spins):
    new_state = spins
    if new_state != '':
        if new_state[-1] == '1' and new_state[0] == '0':
            temp_state = list(new_state)
            temp_state[-1] = '0'
            temp_state[0] = '1'
            new_state = ''.join(temp_state)
        else:
            new_state = ''
    return new_state


def calculate_interaction(spins, is_pbc):
    list_of_spins = []
    for i in range(len(spins) - 1):
        list_of_spins.append(splus_sminus(i, spins))
        list_of_spins.append(sminus_splus(i, spins))
    if is_pbc:
        list_of_spins.append(splus_sminus_boundary(spins))
        list_of_spins.append(sminus_splus_boundary(spins))
    return list_of_spins


def s_z(spin_number, spins):
    spin_value = 0
    if spin_number < len(spins) and spins != '':
        if spins[spin_number] == '1':
            spin_value = 1 / 2
        elif spins[spin_number] == '0':
            spin_value = (-1) * (1 / 2)
    return spin_value


def s_z_total(spins):
    total_spin = 0
    for i in range(len(spins)):
        total_spin += s_z(i, spins)
    return total_spin


def calculate_expected_value(operator, vector):
    left_element = trans(vector)
    expected_value = left_element * (operator * vector)
    try:
        return round(as_scalar(expected_value), 8)
    except TypeError:
        return as_scalar(expected_value)


def calculate_matrix_element(vector_1, operator, vector_2):
    left_vector = trans(cx_mat(vector_1))
    right_vector = cx_mat(vector_2)
    matrix_element = left_vector * (operator * right_vector)
    try:
        return round(as_scalar(matrix_element), 8)
    except TypeError:
        return as_scalar(matrix_element)


def calculate_variation(operator, vector):
    expected_of_square = calculate_expected_value(operator * operator, vector)
    expected_value = calculate_expected_value(operator, vector)
    square_of_expected = pow(expected_value, 2)
    return round(expected_of_square - square_of_expected, 6)


def create_initial_state(size):
    initial_state = ''
    for spin in range(size):
        initial_state += '1' if spin < size / 2 else '0'
    return initial_state


def wave_vector(L, k):
    return 2 * pi * k / L


def convert_string_vector_to_pyarma_vector(string_vector):
    inverted_vector = string_vector[::-1]
    spin_down = mat([0, 1]).t()
    spin_up = mat([1, 0]).t()
    pyarma_vector = spin_down if inverted_vector[0] == '0' else spin_up
    for spin in inverted_vector[1:]:
        if spin == '0':
            pyarma_vector = kron(pyarma_vector, spin_down)
        elif spin == '1':
            pyarma_vector = kron(pyarma_vector, spin_up)
        else:
            pass
    return pyarma_vector
