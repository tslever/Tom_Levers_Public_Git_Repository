'''
Module for class TestAnalyzer, which tests the methods of an Analyzer object
'''

from montecarlosimulator import Die
from montecarlosimulator import Game
from montecarlosimulator import Analyzer
import numpy as np
import pandas as pd
import unittest

class TestAnalyzer(unittest.TestCase):
    '''
    Tests the methods of an Analyzer object

    Instance variables:
        none

    Public methods:
        test_init
    '''

    def test_init(self):
        '''
        Tests Analyzer.__init__

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Compares data frames of rolls and face counts, where the number of rows and observations is the number of rolls, the number of columns and features is the number of faces, and each cell value is a count of the number of dice for one roll with a face

        Exceptions raised:
            AssertionError if a shown data frame of rolls and face counts does not equal an expected data frame of rolls and face counts

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
        analyzer = Analyzer(game)
        data_frame_of_rolls_and_face_counts = analyzer.generate_data_frame_of_rolls_and_face_counts()
        data_frame_of_rolls_and_dice = game.show('wide')
        expected_data_frame_of_rolls_and_face_counts = data_frame_of_rolls_and_dice.apply(lambda series_of_faces: series_of_faces.value_counts(), axis = 1).fillna(0).astype(dtype = np.int8)
        self.assertTrue(data_frame_of_rolls_and_face_counts.equals(expected_data_frame_of_rolls_and_face_counts))
        Die._roll_is_being_tested = False

    def test_generate_data_frame_of_rolls_and_face_counts(self):
        '''
        Tests Analyzer.generate_data_frame_of_rolls_and_face_counts

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Compares data frames of rolls and face counts, where the number of rows and observations is the number of rolls, the number of columns and features is the number of faces, and each cell value is a count of the number of dice for one roll with a face

        Exceptions raised:
            AssertionError if a shown data frame of rolls and face counts does not equal an expected data frame of rolls and face counts

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
        analyzer = Analyzer(game)
        data_frame_of_rolls_and_face_counts = analyzer.generate_data_frame_of_rolls_and_face_counts()
        data_frame_of_rolls_and_dice = game.show('wide')
        expected_data_frame_of_rolls_and_face_counts = data_frame_of_rolls_and_dice.apply(lambda series_of_faces: series_of_faces.value_counts(), axis = 1).fillna(0).astype(dtype = np.int8)
        self.assertTrue(data_frame_of_rolls_and_face_counts.equals(expected_data_frame_of_rolls_and_face_counts))
        Die._roll_is_being_tested = False

    def test_get_number_of_rolls_where_all_dice_have_the_same_face(self):
        '''
        Tests Analyzer.test_get_number_of_rolls_where_all_dice_have_the_same_face

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Compares a number of rolls with a number of faces with counts greater than zero equal to one with an expected number of rolls with a number of faces with counts greater than zero equal to one
            Compares data frames of rolls and face counts where all dice for one roll have the same face

        Exceptions raised:
            AssertionError if a number of rolls with a number of faces with counts greater than zero equal to one is not equal to an expected number of rolls with a number of faces with counts greater than zero equal to one, or
                              a data frame of rolls and face counts where all dice for one roll have the same face are not equal

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
        analyzer = Analyzer(game)
        number_of_rolls_with_number_of_faces_with_counts_greater_than_zero_equal_to_one = analyzer.get_number_of_rolls_where_all_dice_have_the_same_face()
        self.assertEqual(number_of_rolls_with_number_of_faces_with_counts_greater_than_zero_equal_to_one, 20)
        data_frame_of_face_combinations_and_counts_where_combinations_have_all_faces_the_same = analyzer.data_frame_of_face_combinations_and_counts_where_combinations_have_all_faces_the_same
        data_frame_of_face_combinations_and_counts = analyzer.data_frame_of_face_combinations_and_counts
        first_face_combination = data_frame_of_face_combinations_and_counts.index[0]
        number_of_faces = len(first_face_combination)
        list_with_elements_face = ['face'] * number_of_faces
        empty_multiIndex = pd.MultiIndex.from_tuples([], names = list_with_elements_face)
        expected_data_frame_of_face_combinations_and_counts_where_combinations_have_all_faces_the_same = pd.DataFrame(index = empty_multiIndex, columns = ['count'])
        for face_combination, series_of_face_combination_and_count in data_frame_of_face_combinations_and_counts.iterrows():
            set_of_unique_faces = set(face_combination)
            number_of_unique_faces = len(set_of_unique_faces)
            if number_of_unique_faces == 1:
                count = series_of_face_combination_and_count['count']
                expected_data_frame_of_face_combinations_and_counts_where_combinations_have_all_faces_the_same.loc[face_combination, :] = count
        self.assertTrue(data_frame_of_face_combinations_and_counts_where_combinations_have_all_faces_the_same.equals(expected_data_frame_of_face_combinations_and_counts_where_combinations_have_all_faces_the_same))

        list_of_fair_coins = []
        for i in range(0, 3):
            array_of_faces = np.array(['H', 'T'], dtype = str)
            fair_coin = Die(array_of_faces)
            list_of_fair_coins.append(fair_coin)
        game_with_three_fair_coins = Game(list_of_fair_coins)
        game_with_three_fair_coins.play(1000)
        analyzer = Analyzer(game_with_three_fair_coins)
        number_of_jackpots = analyzer.get_number_of_rolls_where_all_dice_have_the_same_face()
        self.assertEqual(number_of_jackpots, 1000)
        data_frame_of_face_combinations_and_counts_where_combinations_have_all_faces_the_same = analyzer.data_frame_of_face_combinations_and_counts_where_combinations_have_all_faces_the_same
        data_frame_of_face_combinations_and_counts = analyzer.data_frame_of_face_combinations_and_counts
        first_face_combination = data_frame_of_face_combinations_and_counts.index[0]
        number_of_faces = len(first_face_combination)
        list_with_elements_face = ['face'] * number_of_faces
        empty_multiIndex = pd.MultiIndex.from_tuples([], names = list_with_elements_face)
        expected_data_frame_of_face_combinations_and_counts_where_combinations_have_all_faces_the_same = pd.DataFrame(index = empty_multiIndex, columns = ['count'])
        for face_combination, series_of_face_combination_and_count in data_frame_of_face_combinations_and_counts.iterrows():
            set_of_unique_faces = set(face_combination)
            number_of_unique_faces = len(set_of_unique_faces)
            if number_of_unique_faces == 1:
                count = series_of_face_combination_and_count['count']
                expected_data_frame_of_face_combinations_and_counts_where_combinations_have_all_faces_the_same.loc[face_combination, :] = count
        self.assertTrue(data_frame_of_face_combinations_and_counts_where_combinations_have_all_faces_the_same.equals(expected_data_frame_of_face_combinations_and_counts_where_combinations_have_all_faces_the_same))
        Die._roll_is_being_tested = False

    def test_generate_data_frame_of_face_combinations_and_counts(self):
        '''
        Tests Analyzer.generate_data_frame_of_face_combinations_and_counts

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Compares data frames of face combinations and counts of how many times each face combination was rolled

        Exceptions raised:
            AssertionError if two data frames of face combinations and counts of how many times each face combination was rolled are not equal

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
        analyzer = Analyzer(game)
        data_frame_of_face_combinations_and_counts = analyzer.generate_data_frame_of_face_combinations_and_counts()
        data_frame_of_rolls_and_dice = game.show('wide')
        number_of_faces = data_frame_of_rolls_and_dice.shape[1]
        list_with_elements_face = ['face'] * number_of_faces
        empty_multiIndex = pd.MultiIndex.from_tuples([], names = list_with_elements_face)
        expected_data_frame_of_face_combinations_and_counts = pd.DataFrame(index = empty_multiIndex, columns = ['count'])
        for roll_index, series_of_faces in data_frame_of_rolls_and_dice.iterrows():
            list_of_faces = series_of_faces.to_list()
            list_of_sorted_faces = sorted(list_of_faces)
            face_combination = tuple(list_of_sorted_faces)
            if expected_data_frame_of_face_combinations_and_counts.index.isin([face_combination]).any():
                expected_data_frame_of_face_combinations_and_counts.at[face_combination, 'count'] += 1
            else:
                expected_data_frame_of_face_combinations_and_counts.at[face_combination, 'count'] = 1
        self.assertTrue(data_frame_of_face_combinations_and_counts.equals(expected_data_frame_of_face_combinations_and_counts))
        Die._roll_is_being_tested = False

    def test_play(self):
        '''
        Tests Analyzer.play

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Compares an indicator of whether a data frame of face combinations and counts needs to be generated when an analyzer is initialized, after the data frame is generated, and when the analyzer plays with an expected indicator

        Exceptions raised:
            AssertionError if an indicator of whether a data frame of face combinations and counts needs to be generated is not equal to an expected indicator

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
        analyzer = Analyzer(game)
        self.assertTrue(analyzer._data_frame_of_face_combinations_and_counts_needs_to_be_generated)
        analyzer.get_number_of_rolls_where_all_dice_have_the_same_face()
        self.assertFalse(analyzer._data_frame_of_face_combinations_and_counts_needs_to_be_generated)
        analyzer.play(1000)
        self.assertTrue(analyzer._data_frame_of_face_combinations_and_counts_needs_to_be_generated)

if __name__ == "__main__":
    verbose = 2
    unittest.main(verbosity = verbose)