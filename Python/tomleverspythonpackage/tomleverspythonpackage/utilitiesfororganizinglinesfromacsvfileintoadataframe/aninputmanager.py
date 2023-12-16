from tomleverspythonpackage.utilitiesfororganizinglinesfromacsvfileintoadataframe.exceptions.acheckcommandlineargumentsexception import *
from tomleverspythonpackage.utilitiesfororganizinglinesfromacsvfileintoadataframe.exceptions.aninvalidlineindicesexception import *


#################################################################################################################################################################################################################
# An_Input_Manager represents the structure for an input manager that has functionality to check the appropriateness of a vector of command-line arguments for organizing lines from a CSV file into a dataframe.
#
# Author: Tom Lever
# Version: 1.0
# Since: 06/24/21
#################################################################################################################################################################################################################

class An_Input_Manager:


    # -------------------------------------------------------------------------------------------------------------------------------------------------------
    # check checks the appropriateness of a vector of command line arguments of command-line arguments for organizing lines from a CSV file into a dataframe.
    # -------------------------------------------------------------------------------------------------------------------------------------------------------

    def checks(self, the_vector_of_command_line_arguments):

        if (not isinstance(the_vector_of_command_line_arguments, list)) or (not all(isinstance(the_argument, str) for the_argument in the_vector_of_command_line_arguments)):
            raise A_Check_Command_Line_Arguments_Exception("An input manager did not receive a vector of strings.")

        THE_NUMBER_OF_COMMAND_LINE_ARGUMENTS = 4

        if len(the_vector_of_command_line_arguments) != THE_NUMBER_OF_COMMAND_LINE_ARGUMENTS:
            raise A_Check_Command_Line_Arguments_Exception( \
                "\nPass interpreter 'python' the name of this script, the path to a CSV file, the index of the first line to read, and the index of the last line to read."
            )

        if not the_vector_of_command_line_arguments[1].isnumeric():
            raise An_Invalid_Line_Indices_Exception("The command-line argument representing the index of the first line to read is not an integer.")

        if not the_vector_of_command_line_arguments[2].isnumeric():
            raise An_Invalid_Line_Indices_Exception("The command-line argument representing the index of the last line to read is not an integer.")

        if int(the_vector_of_command_line_arguments[1]) < 0:
            raise An_Invalid_Line_Indices_Exception("The index of the first line to read is less than 0.")

        if int(the_vector_of_command_line_arguments[1]) > int(the_vector_of_command_line_arguments[2]):
            raise An_Invalid_Line_Indices_Exception("The index of the first line to read is greater than the index of the last line to read.")