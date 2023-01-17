import numpy as np
from montecarlosimulator import Die
array_of_faces = np.array(['A', 'B', 'C'], dtype = str)
die = Die(array_of_faces)
print(die.show())

die.change_weight(face = 'A', weight = 2.0)
print(die.show())

list_of_rolled_faces = die.roll(number_of_rolls = 3)
print(list_of_rolled_faces)