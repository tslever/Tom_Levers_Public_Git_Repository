import argparse

def convert_floating_point_binary_number_to_decimal(string_representation_of_floating_point_binary_number, number_of_bits_in_exponent):
    string_representation_of_sign_bit = string_representation_of_floating_point_binary_number[0]
    sign_bit = int(string_representation_of_sign_bit)
    print(f'sign bit: {sign_bit}')
    string_representation_of_exponent_in_excess_4_notation = string_representation_of_floating_point_binary_number[1 : 1 + number_of_bits_in_exponent]
    exponent = int(string_representation_of_exponent_in_excess_4_notation, 2) - 2**(number_of_bits_in_exponent - 1)
    print(f'exponent: {exponent}')
    mantissa = 0
    string_representation_of_mantissa = string_representation_of_floating_point_binary_number[1 + number_of_bits_in_exponent : ]
    for i, bit in enumerate(string_representation_of_mantissa):
        mantissa += int(bit) * 2**(-1 - i)
    print(f'mantissa: {mantissa}')
    decimal_number = (-1)**sign_bit * 2**exponent * mantissa
    return decimal_number

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'Convert Floating Point Binary Number To Decimal', description = 'This program converts a floating-point binary number to decimal.')
    parser.add_argument('string_representation_of_number', help = 'string representation of number')
    parser.add_argument('initial_format', choices = ['floating-point'], help = 'initial format')
    parser.add_argument('number_of_bits_in_exponent', help = 'number of bits in exponent')
    args = parser.parse_args()
    string_representation_of_number = args.string_representation_of_number
    initial_format = args.initial_format
    number_of_bits_in_exponent = int(args.number_of_bits_in_exponent)
    print(f'string representation of number: {string_representation_of_number}')
    print(f'initial format: {initial_format}')
    print(f'number of bits in exponent: {number_of_bits_in_exponent}')
    decimal_number = convert_floating_point_binary_number_to_decimal(string_representation_of_number, number_of_bits_in_exponent)
    print(f'decimal number: {decimal_number}')