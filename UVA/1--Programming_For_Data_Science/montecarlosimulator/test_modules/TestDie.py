'''
Module for class TestDie, which tests the methods of a Die object
'''

from montecarlosimulator.Die import *
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

if __name__ == "__main__":
    verbose = 2
    unittest.main(verbosity = verbose)