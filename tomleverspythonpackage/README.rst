To install from root of tomleverspythonpackage: pip install .
To install from GitHub: pip install git+ssh://git@github.com/tslever/Tom_Levers_Git_Repository.git@main#subdirectory=tomleverspythonpackage


Regarding creating this package: https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/
PEP 8 - Style Guide for Python Code:  https://peps.python.org/pep-0008/#package-and-module-names


Usage Examples

import tomleverspythonpackage.run_print_hello_world
Hello,World!
from tomleverspythonpackage import run_print_hello_world
Hello, World!

#import tomleverspythonpackage.print_hello_world.print_hello_world # does not work after running "python" in CLI
from tomleverspythonpackage.print_hello_world import print_hello_world
print_hello_world()
Hello, World!

from tomleverspythonpackage.acsvfilereader import ACsvFileReader