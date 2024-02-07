# Modifies pip packages
# Usage:
# python modifypippackages.py
# from tomleverspythonpackage.modifypippackages import update_pip_packages; update_pip_packages()

import argparse
import subprocess

def get_list_of_packages() -> list[str]:
    list_of_packages = []
    subprocess.run(f'pip list --format=freeze > list_of_pip_packages.txt', shell = True)
    with open('list_of_pip_packages.txt', 'r') as file:
        for line in file:
            line = line.split('==')
            line = line[0]
            list_of_packages.append(line)
    return list_of_packages

def delete_pip_packages() -> None:
    for package in get_list_of_packages():
        if (package != 'pip') and (package != 'pipenv'):
            subprocess.run(f'pip uninstall -y {package}', shell = True)

def update_pip_packages() -> None:
    for package in get_list_of_packages():
        subprocess.run(f'pip install --upgrade {package}', shell = True)

def parse_arguments():
    dictionary_of_arguments = {}
    parser = argparse.ArgumentParser(prog = 'Modify pip Packages', description = 'This program modifies pip packages.')
    parser.add_argument('delete_or_update', help = 'delete or update')
    args = parser.parse_args()
    delete_or_update = args.delete_or_update
    print(f'delete or update: {delete_or_update}')
    dictionary_of_arguments['delete_or_update'] = delete_or_update
    return dictionary_of_arguments

if __name__ == '__main__':
    dictionary_of_arguments = parse_arguments()
    delete_or_update = dictionary_of_arguments['delete_or_update']
    if delete_or_update == 'delete':
        delete_pip_packages()
    else:
        update_pip_packages()