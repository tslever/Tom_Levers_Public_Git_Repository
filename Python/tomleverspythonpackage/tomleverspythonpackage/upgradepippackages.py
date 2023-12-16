import subprocess

def upgrade_pip_packages():
    subprocess.run(f'pip list --format=freeze > list_of_pip_packages.txt', shell = True)
    with open('list_of_pip_packages.txt', 'r') as file:
        for line in file:
            line = line.split('==')
            line = line[0]
            subprocess.run(f'pip install --upgrade {line}', shell = True)
