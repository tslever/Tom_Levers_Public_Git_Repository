############################################################################################################################################################
# Utilities_for_organizing_lines_from_a_CSV_file_into_a_dataframe
# A package for organizing lines from a CSV file between a first index and a last index into a dataframe.
#
# Usage:
# python organizelinesfromacsvfileintoadataframe.py Dataset_With_Headers.csv 1 3 None
# from tomleverspythonpackage.utilitiesfororganizinglinesfromacsvfileintoadataframe.organizelinesfromacsvfileintoadataframe import organize_lines_from_a_CSV_file_into_a_data_frame; organize_lines_from_a_CSV_file_into_a_data_frame('Dataset_With_Headers.csv', '1', '3', 'None')
#
# Author: Tom Lever
# Version: 1.0
# Since: 12/16/2023
############################################################################################################################################################

import argparse
from tomleverspythonpackage.utilitiesfororganizinglinesfromacsvfileintoadataframe.acsvfileorganizer import *
from tomleverspythonpackage.utilitiesfororganizinglinesfromacsvfileintoadataframe.aninputmanager import *
from tomleverspythonpackage.utilitiesfororganizinglinesfromacsvfileintoadataframe.exceptions.acheckcommandlineargumentsexception import *
from tomleverspythonpackage.utilitiesfororganizinglinesfromacsvfileintoadataframe.exceptions.afieldvaluesmismatchexception import *
from tomleverspythonpackage.utilitiesfororganizinglinesfromacsvfileintoadataframe.exceptions.aninvalidpathexception import *
from tomleverspythonpackage.utilitiesfororganizinglinesfromacsvfileintoadataframe.exceptions.aninvalidlineindicesexception import *


# main represents the primary function of the "Organize lines from a CSV file into a dataframe" program, which organizes lines in a CSV file between a first
# index and a last index into a dataframe.

def organize_lines_from_a_CSV_file_into_a_data_frame(the_path_to_the_CSV_file_to_use, the_index_of_the_first_line, the_index_of_the_second_line, the_list_of_columns_as_string):

    try:
        An_Input_Manager().checks([the_path_to_the_CSV_file_to_use, the_index_of_the_first_line, the_index_of_the_second_line, the_list_of_columns_as_string])
    except A_Check_Command_Line_Arguments_Exception as the_invalid_number_of_command_line_arguments_exception:
        print(the_invalid_number_of_command_line_arguments_exception)
        exit()
    except An_Invalid_Line_Indices_Exception as the_invalid_line_indices_exception:
        print(the_invalid_line_indices_exception)
        exit()

    try:
        the_CSV_file_organizer = A_CSV_File_Organizer(the_path_to_the_CSV_file_to_use)
    except An_Invalid_Path_Exception as the_invalid_CSV_file_exception:
        print(the_invalid_CSV_file_exception)
        exit()

    try:
        if the_list_of_columns_as_string == 'None':
            list_of_columns = None
        else:
            list_of_columns = the_list_of_columns_as_string.strip("[").strip("]").split(", ")
        the_CSV_file_organizer.provides_a_dataframe_representing_the_lines_of_the_CSV_file(int(the_index_of_the_first_line), int(the_index_of_the_second_line), list_of_columns)
    except An_Invalid_Line_Indices_Exception as the_invalid_line_indices_exception:
        print(the_invalid_line_indices_exception)
        exit()
    except A_Fields_Values_Mismatch_Exception as the_fields_values_mismatch_exception:
        print(the_fields_values_mismatch_exception)
        exit()

def parse_arguments():
    dictionary_of_arguments = {}
    parser = argparse.ArgumentParser(prog = 'Organize Lines From A CSV File Into A Data Frame', description = 'This program organizes lines from a CSV file into a data frame.')
    parser.add_argument('the_path_to_the_CSV_file_to_use', help = 'the path to the CSV file to use')
    parser.add_argument('the_index_of_the_first_line', help = 'the index of the first line')
    parser.add_argument('the_index_of_the_second_line', help = 'the index of the second line')
    parser.add_argument('the_list_of_columns_as_string', help = 'the list of columns as string')
    args = parser.parse_args()
    the_path_to_the_CSV_file_to_use = args.the_path_to_the_CSV_file_to_use
    the_index_of_the_first_line = args.the_index_of_the_first_line
    the_index_of_the_second_line = args.the_index_of_the_second_line
    the_list_of_columns_as_string = args.the_list_of_columns_as_string
    print(f'the path to the CSV file to use: {the_path_to_the_CSV_file_to_use}')
    print(f'the index of the first line: {the_index_of_the_first_line}')
    print(f'the index of the second line: {the_index_of_the_second_line}')
    print(f'the list of columns: {the_list_of_columns_as_string}')
    dictionary_of_arguments['the_path_to_the_CSV_file_to_use'] = the_path_to_the_CSV_file_to_use
    dictionary_of_arguments['the_index_of_the_first_line'] = the_index_of_the_first_line
    dictionary_of_arguments['the_index_of_the_second_line'] = the_index_of_the_second_line
    dictionary_of_arguments['the_list_of_columns_as_string'] = the_list_of_columns_as_string
    return dictionary_of_arguments

if __name__ == "__main__":
    dictionary_of_arguments = parse_arguments()
    organize_lines_from_a_CSV_file_into_a_data_frame(
        the_path_to_the_CSV_file_to_use = dictionary_of_arguments['the_path_to_the_CSV_file_to_use'],
        the_index_of_the_first_line = dictionary_of_arguments['the_index_of_the_first_line'],
        the_index_of_the_second_line = dictionary_of_arguments['the_index_of_the_second_line'],
        the_list_of_columns_as_string = dictionary_of_arguments['the_list_of_columns_as_string']
    )