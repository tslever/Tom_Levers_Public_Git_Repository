# Converts a list of hexademical numbers to a list of decimal numbers
# Usage:
# At command line, enter python convertlistofhexadecimalnumberstolistofdecimalnumbers.py <array of numbers surrounded by double quotes>
# import tomleverspythonpackage; tomleverspythonpackage.convertlistofhexadecimalnumberstolistofdecimalnumbers.translate(['10', '11'])
# import tomleverspythonpackage.convertlistofhexadecimalnumberstolistofdecimalnumbers; tomleverspythonpackage.convertlistofhexadecimalnumberstolistofdecimalnumbers.translate(['10', '11'])
# from tomleverspythonpackage import convertlistofhexadecimalnumberstolistofdecimalnumbers; convertlistofhexadecimalnumberstolistofdecimalnumbers.translate(['10', '11'])

import argparse

def translate(list_of_hexadecimal_numbers_as_strings):
    string = "["
    number_of_hexadecimal_numbers = len(list_of_hexadecimal_numbers_as_strings)
    for i in range(0, number_of_hexadecimal_numbers):
        number_as_string = list_of_hexadecimal_numbers_as_strings[i]
        integer = int("0x" + number_as_string, 16)
        string += str(integer)
        if (i < number_of_hexadecimal_numbers - 1):
            string += ", "
        else:
            string += "]"
    print(string)

def parse_arguments():
    dictionary_of_arguments = {}
    parser = argparse.ArgumentParser(prog = 'Convert List Of Hexadecimal Numbers To List Of Decimal Numbers', description = 'This program converts a list of hexadecimal numbers to a list of decimal numbers.')
    parser.add_argument('list_of_hexadecimal_numbers_as_string', help = 'list of hexadecimal numbers as string')
    args = parser.parse_args()
    list_of_hexadecimal_numbers_as_string = args.list_of_hexadecimal_numbers_as_string
    print(f'list of hexadecimal numbers as string: {list_of_hexadecimal_numbers_as_string}')
    list_of_tokens = list_of_hexadecimal_numbers_as_string.strip("[]").split()
    list_of_hexadecimal_numbers_as_strings = [token.strip(",") for token in list_of_tokens]
    dictionary_of_arguments['list_of_hexadecimal_numbers_as_strings'] = list_of_hexadecimal_numbers_as_strings
    return dictionary_of_arguments

if __name__ == "__main__":
    dictionary_of_arguments = parse_arguments()
    translate(dictionary_of_arguments['list_of_hexadecimal_numbers_as_strings'])