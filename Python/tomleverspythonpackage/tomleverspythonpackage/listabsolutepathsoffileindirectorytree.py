import os
import sys

# Example directory tree: '/Users/tlever/Desktop/documents_from_DOI'
def list_absolute_paths_of_file_in_directory_tree(path_to_root_of_directory_tree):
    generator = os.walk(path_to_root_of_directory_tree)
    list_of_paths_of_file = []
    for path_of_parent_directory, list_of_names_of_directory, list_of_names_of_file in generator:
        for name_of_file in list_of_names_of_file:
            list_of_paths_of_file.append(path_of_parent_directory + '/' + name_of_file)
    return list_of_paths_of_file

if __name__ == '__main__':
    path_to_root_of_directory_tree = sys.argv[1]
    list_absolute_paths_of_file_in_directory_tree(path_to_root_of_directory_tree)