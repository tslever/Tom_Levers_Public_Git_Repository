import numpy as np
from montecarlosimulator import Die
array_of_faces = np.array(['A', 'B', 'C'], dtype = str)
die = Die(array_of_faces)
print(die.show())

die.change_weight(face = 'A', weight = 2.0)
print(die.show())

list_of_rolled_faces = die.roll(number_of_rolls = 3)
print(list_of_rolled_faces)

die_2 = Die(array_of_faces)
print(die_2.show())

list_of_dice = [die, die_2]
from montecarlosimulator import Game
game = Game(list_of_dice)
game.play(number_of_rolls = 1000)
print(game.show(form = 'wide'))

print(game.show(form = 'narrow'))

from montecarlosimulator import Analyzer
analyzer = Analyzer(game)
data_frame_of_rolls_and_face_counts = analyzer.generate_data_frame_of_rolls_and_face_counts()
print(data_frame_of_rolls_and_face_counts)

number_of_rolls_where_all_dice_have_the_same_face = analyzer.get_number_of_rolls_where_all_dice_have_the_same_face()
print(number_of_rolls_where_all_dice_have_the_same_face)

data_frame_of_face_combinations_and_counts = analyzer.generate_data_frame_of_face_combinations_and_counts()
print(data_frame_of_face_combinations_and_counts)

analyzer.play(1000)
data_frame_of_rolls_and_face_counts = analyzer.generate_data_frame_of_rolls_and_face_counts()
print(data_frame_of_rolls_and_face_counts)