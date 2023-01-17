'''
Module for class TestGame, which tests the methods of a Game object
'''

from montecarlosimulator import Die
from montecarlosimulator import Game
import numpy as np
import pandas as pd
import unittest

class TestGame(unittest.TestCase):
    '''
    Tests the methods of a Game object

    Instance variables:
        none

    Public methods:
        test_init
        test_play
        test_show
    '''

    def test_init(self):
        '''
        Tests Game.__init__

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Compares data frames of rolls and dice, where each number of rows and observations is the number of rolls, each number of columns and features is the number of dice, and each cell value is a face rolled

        Exceptions raised:
            AssertionError if a shown data frame of rolls and dice does not equal an expected data frame of rolls and dice

        Restrictions on when this method can be called:
            none
        '''

        Die._roll_is_being_tested = True
        list_of_dice = []
        array_of_faces = np.array([1, 2, 3, 4], dtype = np.int8)
        for i in range(0, 10):
            die = Die(array_of_faces)
            list_of_dice.append(die)
        game = Game(list_of_dice)
        game.play(20)
        shown_data_frame_of_rolls_and_dice = game.show('wide')
        expected_data_frame_of_rolls_and_dice = pd.DataFrame()
        expected_list_of_rolled_faces = [3, 3, 3, 3, 2, 3, 2, 4, 4, 2, 4, 3, 3, 4, 1, 1, 1, 4, 4, 4]
        for i in range(0, 10):
            expected_data_frame_of_rolls_and_dice[i] = expected_list_of_rolled_faces
        self.assertTrue(shown_data_frame_of_rolls_and_dice.equals(shown_data_frame_of_rolls_and_dice))
        Die._roll_is_being_tested = False

    def test_play(self):
        '''
        Tests Game.play

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Compares data frames of rolls and dice, where each number of rows and observations is the number of rolls, each number of columns and features is the number of dice, and each cell value is a face rolled

        Exceptions raised:
            AssertionError if a shown data frame of rolls and dice does not equal an expected data frame of rolls and dice

        Restrictions on when this method can be called:
            none
        '''
        
        Die._roll_is_being_tested = True
        list_of_dice = []
        array_of_faces = np.array([1, 2, 3, 4], dtype = np.int8)
        for i in range(0, 10):
            die = Die(array_of_faces)
            list_of_dice.append(die)
        game = Game(list_of_dice)
        game.play(20)
        shown_data_frame_of_rolls_and_dice = game.show('wide')
        expected_data_frame_of_rolls_and_dice = pd.DataFrame()
        expected_list_of_rolled_faces = [3, 3, 3, 3, 2, 3, 2, 4, 4, 2, 4, 3, 3, 4, 1, 1, 1, 4, 4, 4]
        for i in range(0, 10):
            expected_data_frame_of_rolls_and_dice[i] = expected_list_of_rolled_faces
        self.assertTrue(shown_data_frame_of_rolls_and_dice.equals(shown_data_frame_of_rolls_and_dice))
        Die._roll_is_being_tested = False

    def test_show(self):
        '''
        Tests Game.show

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Compares data frames of rolls and dice with integer and string faces, where each number of rows and observations is the number of rolls, each number of columns and features is the number of dice, and each cell value is a face rolled.
            Compares data frames of rolls, dice, and integer and string faces, where each row corresponds to a roll, each data frame has a face column, and each cell value is a face rolled.

        Exceptions raised:
            AssertionError if a shown data frame of rolls and dice does not equal an expected data frame of rolls and dice, or
                              a shown data frame of rolls and dice and faces does not equal an expected data frame of rolls and dice and faces

        Restrictions on when this method can be called:
            none
        '''

        Die._roll_is_being_tested = True
        list_of_dice = []
        array_of_faces = np.array([1, 2, 3, 4], dtype = np.int8)
        for i in range(0, 10):
            die = Die(array_of_faces)
            list_of_dice.append(die)
        game = Game(list_of_dice)
        game.play(20)
        shown_data_frame_of_rolls_and_dice = game.show('wide')
        expected_data_frame_of_rolls_and_dice = pd.DataFrame()
        expected_list_of_rolled_faces = [3, 3, 3, 3, 2, 3, 2, 4, 4, 2, 4, 3, 3, 4, 1, 1, 1, 4, 4, 4]
        for i in range(0, 10):
            expected_data_frame_of_rolls_and_dice[i] = expected_list_of_rolled_faces
        self.assertTrue(shown_data_frame_of_rolls_and_dice.equals(shown_data_frame_of_rolls_and_dice))
        shown_data_frame_of_rolls_and_dice = game.show('narrow')
        expected_data_frame_of_rolls_and_dice = expected_data_frame_of_rolls_and_dice.stack().to_frame('face')
        expected_data_frame_of_rolls_and_dice.index.rename(['roll_index', 'die_index'], inplace = True)
        self.assertTrue(shown_data_frame_of_rolls_and_dice.equals(shown_data_frame_of_rolls_and_dice))

        list_of_fair_coins = []
        for i in range(0, 3):
            array_of_faces = np.array(['H', 'T'], dtype = str)
            fair_coin = Die(array_of_faces)
            list_of_fair_coins.append(fair_coin)
        game_with_three_fair_coins = Game(list_of_fair_coins)
        number_of_rolls = 10
        game_with_three_fair_coins.play(number_of_rolls)
        shown_data_frame_of_rolls_and_dice = game_with_three_fair_coins.show('wide')
        expected_data_frame_of_rolls_and_dice = pd.DataFrame()
        expected_list_of_rolled_faces = ['T', 'T', 'T', 'T', 'H', 'T', 'H', 'T', 'T', 'H']
        for i in range(0, 3):
            expected_data_frame_of_rolls_and_dice[i] = expected_list_of_rolled_faces
        self.assertTrue(shown_data_frame_of_rolls_and_dice.equals(shown_data_frame_of_rolls_and_dice))
        shown_data_frame_of_rolls_and_dice = game_with_three_fair_coins.show('narrow')
        expected_data_frame_of_rolls_and_dice = expected_data_frame_of_rolls_and_dice.stack().to_frame('face')
        expected_data_frame_of_rolls_and_dice.index.rename(['roll_index', 'die_index'], inplace = True)
        self.assertTrue(shown_data_frame_of_rolls_and_dice.equals(shown_data_frame_of_rolls_and_dice))
        Die._roll_is_being_tested = False

if __name__ == "__main__":
    verbose = 2
    unittest.main(verbosity = verbose)