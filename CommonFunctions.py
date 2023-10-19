from numpy import polyfit, linspace, poly1d, abs
import os
import json
from datetime import datetime


def generate_binary_strings(bit_count):
    binary_strings = []

    def gen_bin(n, bs=''):
        if len(bs) == n:
            binary_strings.append(bs)
        else:
            gen_bin(n, bs + '0')
            gen_bin(n, bs + '1')

    gen_bin(bit_count)
    return binary_strings


def create_ascending_list(quantity):
    return [number + 2 for number in range(quantity-1)]


def delete_odd_numbers(list_of_numbers):
    return [number for number in list_of_numbers if number % 2 == 0]


def extrapolate_data(x_axis_list, y_axis_list):
    margin = (max(x_axis_list) - min(x_axis_list)) / 5
    extrapolated_x_axis = linspace(min(x_axis_list) - margin, max(x_axis_list) + margin, 1000)
    polynomial_coefficients = polyfit(x_axis_list, y_axis_list, 2)
    polynomial_coefficients = [round(coeff, 2) for coeff in polynomial_coefficients]
    polynomial_function = poly1d(polynomial_coefficients)
    extrapolated_y_axis = [polynomial_function(x) for x in extrapolated_x_axis]
    return extrapolated_x_axis, extrapolated_y_axis, polynomial_coefficients


def polynomial_function_string(list_of_coefficients):
    polynomial = ''
    degree = len(list_of_coefficients) - 1
    for coefficient in list_of_coefficients:
        if degree == 0:
            polynomial += f'{str(coefficient)}'
        else:
            polynomial += f'{str(coefficient)}x^{degree} + '
        degree = degree - 1
    return polynomial


def check_if_file_is_empty(file_path):
    return os.stat(file_path).st_size == 0


def check_if_file_exists(file_path):
    return os.path.exists(file_path)


def check_if_file_has_data(file_path, sub_dir=''):
    path = os.path.join(sub_dir, file_path) if sub_dir else file_path
    if check_if_file_exists(path):
        if check_if_file_is_empty(path) is False:
            return True
        else:
            raise FileNotFoundError
    else:
        return False


def make_directories(list_of_dirs):
    for directory in list_of_dirs:
        if not check_if_file_exists(directory):
            os.mkdir(directory)


def ask_to_replace_file():
    print('File with that name already exists. '
          'Do you want to perform new simulation? [y/n]')
    while True:
        answer = input()
        if answer == 'y':
            return True
        elif answer == 'n':
            return False
        else:
            print('Please type "y" for yes or "n" for no.')


def find_nearest_value(np_array, searched_value):
    return (abs(np_array - searched_value)).argmin()


def dirac_delta_function(x,  a, accuracy):
    if -accuracy / 2 <= x - a <= accuracy / 2:
        return 1
    else:
        return 0


def square_of_complex_modulus(complex_number):
    square_modulus = complex_number * complex_number.conjugate()
    return square_modulus.real


def save_json_file(dict_with_data, json_file_name, sub_dir=''):
    path = os.path.join(sub_dir, json_file_name) if sub_dir else json_file_name
    json_file = open(path, 'w')
    json.dump(dict_with_data, json_file)
    json_file.close()
    print(f'Created new file {json_file_name} with simulation data.')


def read_json_file(json_file_name, sub_dir=''):
    path = os.path.join(sub_dir, json_file_name) if sub_dir else json_file_name
    print(f'Reading data from file {json_file_name}...')
    json_file = open(path, 'r')
    dict_with_data = json_file.read()
    dict_with_data = json.loads(dict_with_data)
    json_file.close()
    return dict_with_data


def print_and_store(variable, message=None, disable_print=False, end='\n'):
    if message is None:
        message = f'Program executed on {datetime.now()}.'
        variable += message + end
    else:
        variable += message + end
    if not disable_print:
        print(message, end=end)
    return variable
