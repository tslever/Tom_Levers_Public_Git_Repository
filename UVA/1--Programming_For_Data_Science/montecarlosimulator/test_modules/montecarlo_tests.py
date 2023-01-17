'''
Module for classes TestDie, which tests the methods of a Die object;
TestGame, which tests the methods of a Game object; and
TestAnalyzer, which tests the methods of an Analyzer object
'''

from montecarlosimulator import Die
import numpy as np
import pandas as pd
import unittest

class TestDie(unittest.TestCase):
    '''
    Tests the methods of a Die object

    Instance variables:
        none

    Public methods:
        test_init
        test_change_weight
        test_roll
        test_show
    '''

    def test_init(self):
        '''
        Tests Die.__init__

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Compares data frames of faces and weights for two numpy arrays of faces

        Exceptions raised:
            AssertionError if a shown data frame of faces and weights does not equal an expected data frame of faces and weights

        Restrictions on when this method can be called:
            none
        '''

        array_of_faces = np.array([1, 2, 3, 4], dtype = np.int8)
        array_of_weights = np.ones(len(array_of_faces))
        expected_data_frame_of_faces_and_weights = pd.DataFrame({'face': array_of_faces, 'weight': array_of_weights})
        die = Die(array_of_faces)
        shown_data_frame_of_faces_and_weights = die.show()
        self.assertTrue(shown_data_frame_of_faces_and_weights.equals(expected_data_frame_of_faces_and_weights))

        array_of_faces = np.array(['1', '2', '3', '4'], dtype = str)
        array_of_weights = np.ones(len(array_of_faces))
        expected_data_frame_of_faces_and_weights = pd.DataFrame({'face': array_of_faces, 'weight': array_of_weights})
        die = Die(array_of_faces)
        shown_data_frame_of_faces_and_weights = die.show()
        self.assertTrue(shown_data_frame_of_faces_and_weights.equals(expected_data_frame_of_faces_and_weights))

    def test_change_weight(self):
        '''
        Tests Die.change_weight

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Ensures a weight in the data frame of faces and weights of a die is changed,
                    attempting to change the weight corresponding to a face that does not exist in the data frame of faces and weights of a die raises a value error, and
                    attempting to change a weight to a value that cannot be converted to np.float64 raises a value error

        Exceptions raised:
            AssertionError if a weight in the data frame of faces and weights of a die is not changed,
                              attempting to change the weight corresponding to a face that does not exist in the data frame of faces and weights of a die succeeds, or
                              attempting to change a weight to a value that cannot be converted to np.float64 succeeds

        Restrictions on when this method can be called:
            none
        '''

        array_of_faces = np.array([1, 2, 3, 4], dtype = np.int8)
        array_of_weights = np.ones(len(array_of_faces))
        array_of_weights[0] = 2.0
        expected_data_frame_of_faces_and_weights = pd.DataFrame({'face': array_of_faces, 'weight': array_of_weights})
        die = Die(array_of_faces)
        die.change_weight(1, 2.0)
        shown_data_frame_of_faces_and_weights = die.show()
        self.assertTrue(shown_data_frame_of_faces_and_weights.equals(expected_data_frame_of_faces_and_weights))

        try:
            die.change_weight('face_that_does_not_exist_in_data_frame_of_faces_and_weights_of_die', 2.0)
            self.fail()
        except ValueError as e:
            pass

        try:
            die.change_weight(1, 'weight_that_cannot_be_converted_to_np.float64')
            self.fail()
        except ValueError as e:
            pass

    def test_roll(self):
        '''
        Tests Die.roll

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Ensures a list of rolled faces for a die is equal to an expected list of rolled faces when roll is being tested

        Exceptions raised:
            AssertionError if a list of rolled faces for a die is not equal to an expected list of rolled faces when roll is being tested

        Restrictions on when this method can be called:
            none
        '''
        
        Die._roll_is_being_tested = True
        array_of_faces = np.array([1, 2, 3, 4], dtype = np.int8)
        die = Die(array_of_faces)
        list_of_rolled_faces = die.roll(20)
        expected_list_of_rolled_faces = [3, 3, 3, 3, 2, 3, 2, 4, 4, 2, 4, 3, 3, 4, 1, 1, 1, 4, 4, 4]
        self.assertEqual(list_of_rolled_faces, expected_list_of_rolled_faces)

        array_of_faces = np.array(['H', 'T'], dtype = str)
        die = Die(array_of_faces)
        list_of_rolled_faces = die.roll(10000)
        number_of_heads = list_of_rolled_faces.count('H')
        self.assertEqual(5064, number_of_heads)

        die.change_weight('H', 5.0)
        list_of_rolled_faces = die.roll(10000)
        number_of_heads = list_of_rolled_faces.count('H')
        self.assertEqual(8345, number_of_heads)
        Die._roll_is_being_tested = False

    def test_show(self):
        '''
        Tests Die.show

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Compares data frames of faces and weights for numpy arrays of integer and string faces

        Exceptions raised:
            AssertionError if a shown data frame of faces and weights does not equal an expected data frame of faces and weights

        Restrictions on when this method can be called:
            none
        '''

        array_of_faces = np.array([1, 2, 3, 4], dtype = np.int8)
        array_of_weights = np.ones(len(array_of_faces))
        expected_data_frame_of_faces_and_weights = pd.DataFrame({'face': array_of_faces, 'weight': array_of_weights})
        die = Die(array_of_faces)
        shown_data_frame_of_faces_and_weights = die.show()
        self.assertTrue(shown_data_frame_of_faces_and_weights.equals(expected_data_frame_of_faces_and_weights))

        array_of_faces = np.array(['H', 'T'], dtype = str)
        fair_coin = Die(array_of_faces)
        shown_data_frame_of_faces_and_weights = fair_coin.show()
        array_of_weights = np.ones(len(array_of_faces))
        expected_data_frame_of_faces_and_weights = pd.DataFrame({'face': array_of_faces, 'weight': array_of_weights})
        self.assertTrue(shown_data_frame_of_faces_and_weights.equals(expected_data_frame_of_faces_and_weights))

from montecarlosimulator import Game

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

from montecarlosimulator import Analyzer

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