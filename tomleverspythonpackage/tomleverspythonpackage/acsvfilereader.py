class ACsvFileReader:
    
    def __init__(self) -> None:
        pass

    def reads_into_a_dictionary(self, path: str) -> dict:
        the_dictionary = {
                
        }
        
        with open(path, mode='r') as the_csv_file:
            for line in the_csv_file:
                the_stripped_line = line.rstrip('\n')
                the_list_of_the_key_and_the_value = the_stripped_line.split(',')
                the_key = the_list_of_the_key_and_the_value[0]
                the_value = the_list_of_the_key_and_the_value[1]
                the_dictionary[the_key] = the_value
            return the_dictionary
