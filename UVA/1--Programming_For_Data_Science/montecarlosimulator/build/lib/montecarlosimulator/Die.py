'''
Module for class Die, which may be rolled to select a face
'''

import numpy as np
import pandas as pd
import random

class Die:
    '''
    May be rolled to select a face.
    Has N sides.
    Each side is associated with a face and a weight.
    Weights default to 1.0 and may be changed.

    Class variables:
        _roll_is_being_tested: bool -- an indicator of whether roll is being tested

    Instance variables:
        _data_frame_of_faces_and_weights: pd.DataFrame -- a data frame with an index of faces and a column of weights. A face has a data type of str, int, or float. All faces in the column named 'face' have the same data type. The faces must be unique. A weight has a data type of float.

    Public methods:
        __init__
        change_weight
        roll
        show
    '''

    _roll_is_being_tested = False

    def __init__(self, array_of_faces):
        '''
        Initializes a Die object
        
        Keyword arguments:
            array_of_faces: np.ndarray -- a 1D numpy array of faces. A face must have a data type of str, int, or float. All faces in the numpy array have the same data type. The faces in the numpy array must be unique.

        Return values:
            none

        Side effects:
            Initializes this Die object's data frame of faces and weights

        Exceptions raised:
            none

        Restrictions on when this method can be called:
            May not be called directly
        '''

        array_of_weights = np.ones(len(array_of_faces))
        self._data_frame_of_faces_and_weights = pd.DataFrame({'face': array_of_faces, 'weight': array_of_weights})

    def change_weight(self, face, weight):
        '''
        Changes the weight of a provided face to a provided weight

        Keyword arguments:
            face: str, int, np.float64 -- A string, integer, or floating-point number
            weight: np.float64 -- A numpy floating-point number

        Return values:
            none

        Side effects:
            Changes the weight of the provided face to the provided weight

        Exceptions raised:
            ValueError, if the provided face is not in the index of this Die object's data frame of faces and weights, or if the provided weight cannot be converted to a np.float64 object

        Restrictions on when this method can be called:
            none
        '''

        if not face in self._data_frame_of_faces_and_weights['face'].to_list():
            raise ValueError('face is not in index of faces')
        # Checks to see if weight can be converted to np.float64.
        # If not, raises ValueError: could not convert <weight type> to float: <weight value>
        weight = np.float64(weight)
        mask_face_column_equals_face = self._data_frame_of_faces_and_weights['face'] == face
        index_of_row_with_face = self._data_frame_of_faces_and_weights.index[mask_face_column_equals_face][0]
        self._data_frame_of_faces_and_weights.at[index_of_row_with_face, 'weight'] = weight

    def roll(self, number_of_rolls = 1):
        '''
        Rolls this Die object one or more times

        Keyword arguments:
            number_of_rolls: int -- An integer. Defaults to 1.

        Return values:
            list_of_rolled_faces: list -- A list of rolled faces

        Side effects:
            Samples rows from the data frame of faces and weights of this Die object according to the weights
        
        Exceptions raised:
            none

        Restrictions on when this method can be called:
            none
        '''
        
        the_random_state = 0 if Die._roll_is_being_tested else random.randint(1, 2000000)
        if self._data_frame_of_faces_and_weights.shape[0] > 0:
            type_of_face = type(self._data_frame_of_faces_and_weights.at[0, 'face'])
        else:
            type_of_face = None
        data_frame_of_rolled_faces_and_weights = self._data_frame_of_faces_and_weights.sample(n = number_of_rolls, replace = True, weights = 'weight', random_state = the_random_state, axis = 0).reset_index(drop = True)
        list_of_rolled_faces = [type_of_face(element) for element in data_frame_of_rolled_faces_and_weights['face'].to_list()]
        #list_of_faces = self._data_frame_of_faces_and_weights['face'].to_list()
        #list_of_weights = self._data_frame_of_faces_and_weights['weight'].to_list()
        #list_of_rolled_faces = type_of_face(random.choices(list_of_faces, weights = list_of_weights, k = number_of_rolls))
        return list_of_rolled_faces

    def show(self):
        '''
        Displays and provides the data frame of faces and weights of this Die object

        Keyword arguments:
            none

        Return values:
            _data_frame_of_faces_and_weights: pd.DataFrame -- The data frame of faces and weights of this Die object

        Side effects:
            Displays the data frame of faces and weights of this Die object

        Exceptions raised:
            none

        Restrictions on when this method can be called:
            none
        '''

        #print(self._data_frame_of_faces_and_weights)
        return self._data_frame_of_faces_and_weights