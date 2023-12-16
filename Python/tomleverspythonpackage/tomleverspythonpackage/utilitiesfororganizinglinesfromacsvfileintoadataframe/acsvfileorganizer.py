from .exceptions.afieldvaluesmismatchexception import *
from .exceptions.aninvalidpathexception import *
from .exceptions.aninvalidlineindicesexception import *
import pandas
from pathlib import Path


#################################################################################################################################################################################################################
# A_CSV_File_Organizer represents the structure for a CSV-file organizer that is based on a CSV file and that has functionality to provide a dataframe representing the lines of the CSV file inclusively between
# a first index and a last index.
#
# Author: Tom Lever
# Version: 1.0
# Since: 06/24/21
#################################################################################################################################################################################################################

class A_CSV_File_Organizer:

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # __init__ is the two-parameter constructor for A_CSV_File_Organizer, which sets this CSV-file organizer's path to a CSV file to a provided path. If the provided path is not to a CSV file, an invalid path
    # exception is raised.
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, the_path_to_the_CSV_file_to_use: str):

        if (not Path(the_path_to_the_CSV_file_to_use).is_file()) or (Path(the_path_to_the_CSV_file_to_use).suffix != ".csv"):
            raise An_Invalid_Path_Exception(the_path_to_the_CSV_file_to_use + " is not a CSV file.")

        self.the_path_to_the_CSV_file = the_path_to_the_CSV_file_to_use


    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # provides_a_dataframe_representing_the_lines_of_the_CSV_file_inclusively_between provides a dataframe representing the lines of the CSV file at the path of this CSV-file organizer between a first index
    # and a last index.
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def provides_a_dataframe_representing_the_lines_of_the_CSV_file(self, the_index_of_the_first_line: int, the_index_of_the_second_line: int, the_list_of_columns = None) -> pandas.DataFrame:

        with open(self.the_path_to_the_CSV_file) as the_CSV_file:

            the_list_of_fields = the_CSV_file.readline().strip("\n").split(",")
            if the_list_of_columns is None:
                the_growing_dataframe = pandas.DataFrame(columns = the_list_of_fields)
            else:
                the_growing_dataframe = pandas.DataFrame(columns = the_list_of_columns)
                the_growing_dataframe.to_csv('Data_Frame.csv', mode = 'w', index = False, header = True)
            the_CSV_file.seek(0)

            for i in range(0, the_index_of_the_first_line):
                the_present_line = the_CSV_file.readline()
                if (the_present_line == ""):
                    raise An_Invalid_Line_Indices_Exception("A CSV-file reader found the end of a file.")

            for i in range(the_index_of_the_first_line, the_index_of_the_second_line + 1):

                if i % 1000 == 0:
                    print(i)
                
                the_present_line = the_CSV_file.readline()
                if (the_present_line == ""):
                    raise An_Invalid_Line_Indices_Exception("A CSV-file reader found the end of a file.")
                
                the_list_of_values = the_present_line.strip("\n").split(",")
                if len(the_list_of_values) < len(the_list_of_fields):
                    raise A_Fields_Values_Mismatch_Exception("The number of values in a list does not match the number of fields in a list.")
                elif len(the_list_of_values) > len(the_list_of_fields):
                    the_list_of_values = the_list_of_values[0:len(the_list_of_fields)]

                the_dataframe_representing_the_lists_of_fields_and_values = pandas.DataFrame([the_list_of_values], columns = the_list_of_fields)
                if the_list_of_columns is not None:
                    the_dataframe_representing_the_lists_of_fields_and_values = the_dataframe_representing_the_lists_of_fields_and_values[the_list_of_columns]
                the_dataframe_representing_the_lists_of_fields_and_values.to_csv('Data_Frame.csv', mode = 'a', index = False, header = False)