class CsvFileReader:
    
    def __init__(self) -> None:
        pass

    def reads_into_a_dictionary(self, path: str, cast_key, cast_value) -> dict:
        the_dictionary = {
            
        }
        with open(path, mode='r') as the_csv_file:
            for line in the_csv_file:
                the_stripped_line = line.rstrip('\n')
                the_list_of_the_key_and_the_value_to_add = the_stripped_line.split(',')
                the_key = cast_key(the_list_of_the_key_and_the_value_to_add[0])
                the_value_to_add = cast_value(the_list_of_the_key_and_the_value_to_add[1])
                if the_key not in the_dictionary.keys():
                    the_dictionary[the_key] = the_value_to_add
                else:
                    the_value_that_exists = the_dictionary[the_key]
                    if isinstance(the_value_that_exists, list):
                        the_value_that_exists.append(the_value_to_add)
                    else:
                        the_dictionary[the_key] = [the_value_that_exists, the_value_to_add]
            return the_dictionary