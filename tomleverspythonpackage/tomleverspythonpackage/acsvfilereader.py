class ACsvFileReader:
    
    def __init__(self) -> None:
        pass

    def reads_into_a_dictionary(self, path, caster_for_key, caster_for_value) -> dict:
        the_dictionary = {
                
        }
        with open(path, mode='r') as the_csv_file:
            for line in the_csv_file:
                the_stripped_line = line.rstrip('\n')
                the_list_of_the_key_and_the_value = the_stripped_line.split(',')
                the_key_as_a_string = the_list_of_the_key_and_the_value[0]
                the_value_as_a_string = the_list_of_the_key_and_the_value[1]
                the_key = caster_for_key(the_key_as_a_string)
                the_value = caster_for_value(the_value_as_a_string)
                the_dictionary[the_key] = the_value
            return the_dictionary
