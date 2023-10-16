############################################################################################################################################################
# Utilities_for_organizing_lines_from_a_CSV_file_into_a_dataframe
# A package for organizing lines from a CSV file between a first index and a last index into a dataframe.
#
#Usage: At command line, enter on one line
#python organizelinesfromacsvfileintoadataframe.py
#<the path to the CSV file> <the index of the first line to read> <the index of the last line to read>
#
#Example Usage: At command line, enter on one line
#python organizelinesfromacsvfileintoadataframe.py ../the_sibling_folder/the_CSV_file.csv 1000 1002
#
# Author: Tom Lever
# Version: 1.0
# Since: 06/24/21
############################################################################################################################################################

import sys
from acsvfileorganizer import *
from aninputmanager import *
from exceptions.acheckcommandlineargumentsexception import *
from exceptions.afieldvaluesmismatchexception import *
from exceptions.aninvalidpathexception import *
from exceptions.aninvalidlineindicesexception import *


# main represents the primary function of the "Organize lines from a CSV file into a dataframe" program, which organizes lines in a CSV file between a first
# index and a last index into a dataframe.

def main():

    try:
        An_Input_Manager().checks(sys.argv)
    except A_Check_Command_Line_Arguments_Exception as the_invalid_number_of_command_line_arguments_exception:
        print(the_invalid_number_of_command_line_arguments_exception)
        exit()
    except An_Invalid_Line_Indices_Exception as the_invalid_line_indices_exception:
        print(the_invalid_line_indices_exception)
        exit()

    try:
        the_CSV_file_reader = A_CSV_File_Organizer(sys.argv[1])
    except An_Invalid_Path_Exception as the_invalid_CSV_file_exception:
        print(the_invalid_CSV_file_exception)
        exit()

    try:
        print(the_CSV_file_reader.provides_a_dataframe_representing_the_lines_of_the_CSV_file_inclusively_between(int(sys.argv[2]), int(sys.argv[3])))
    except An_Invalid_Line_Indices_Exception as the_invalid_line_indices_exception:
        print(the_invalid_line_indices_exception)
        exit()
    except A_Fields_Values_Mismatch_Exception as the_fields_values_mismatch_exception:
        print(the_fields_values_mismatch_exception)
        exit()


if __name__ == "__main__":

    main()