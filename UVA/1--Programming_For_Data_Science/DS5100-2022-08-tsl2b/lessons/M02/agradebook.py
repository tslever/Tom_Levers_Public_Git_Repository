class AGradeBook:
    
    def __init__(self, the_gradebook_to_use: dict) -> None:
        self.gradebook = the_gradebook_to_use

    def provides_a_grade_for(self, name: str) -> int:
        list_of_gradebook_keys = list(self.gradebook.keys())
        list_of_gradebook_values = list(self.gradebook.values())
        index_of_name_in_list_of_gradebook_keys = list_of_gradebook_keys.index(name)
        print()
        return list_of_gradebook_values[index_of_name_in_list_of_gradebook_keys]