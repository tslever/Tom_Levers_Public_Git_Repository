# Lists absolute paths of file in directory tree
# Usage:
# python listabsolutepathsoffileindirectorytree.py .
# from tomleverspythonpackage.listabsolutepathsoffileindirectorytree import list_absolute_paths_of_file_in_directory_tree; list_absolute_paths_of_file_in_directory_tree('.')

import argparse
import os

def list_absolute_paths_of_file_in_directory_tree(path_to_root_of_directory_tree):
    generator = os.walk(path_to_root_of_directory_tree)
    list_of_paths_of_file = []
    for path_of_parent_directory, list_of_names_of_directory, list_of_names_of_file in generator:
        for name_of_file in list_of_names_of_file:
            list_of_paths_of_file.append(path_of_parent_directory + '/' + name_of_file)
    print(list_of_paths_of_file)

def parse_arguments():
    dictionary_of_arguments = {}
    parser = argparse.ArgumentParser(prog = 'List Absolute Paths Of File In Directory Tree', description = 'This program lists absolute paths of file in directory tree.')
    parser.add_argument('path_to_root_of_directory_tree', help = 'path_to_root_of_directory_tree')
    args = parser.parse_args()
    path_to_root_of_directory_tree = args.path_to_root_of_directory_tree
    print(f'path to root of directory tree: {path_to_root_of_directory_tree}')
    dictionary_of_arguments['path_to_root_of_directory_tree'] = path_to_root_of_directory_tree
    return dictionary_of_arguments

if __name__ == '__main__':
    dictionary_of_arguments = parse_arguments()
    list_absolute_paths_of_file_in_directory_tree(dictionary_of_arguments['path_to_root_of_directory_tree'])