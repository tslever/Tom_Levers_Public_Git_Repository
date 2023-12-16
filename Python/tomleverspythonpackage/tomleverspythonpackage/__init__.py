'''
tomleverspythonpackage

A Python package offering Tom Lever's Python solutions

Exports:
    remove_input
'''

__version__ = "0.1.0"
__author__ = "Tom Lever"
__credits__ = "Tom Lever"

from .convertlistofhexadecimalnumberstolistofdecimalnumbers import * # Allows import tomleverspythonpackage; tomleverspythonpackage.convertlistofhexadecimalnumberstolistofdecimalnumbers.translate(['10', '11'])
from .csvfilereader import *
from .exporttopdf import *
from .getlistofanchortagswithhyperlinkreferences import *
from .joindatasets import *
from .listabsolutepathsoffileindirectorytree import *
from .upgradeallpippackages import *
from .utilitiesfororganizinglinesfromacsvfileintoadataframe import *