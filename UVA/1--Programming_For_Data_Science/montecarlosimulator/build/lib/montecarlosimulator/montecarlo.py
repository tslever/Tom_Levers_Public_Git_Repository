'''
Module for classes Die, which may be rolled to select a face;
Game, which plays by rolling one or more times all dice in a list of one or more dice with the same set of faces; and
Analyzer, which generates structures of descriptive statistics for a game that has been played
'''

import numpy as np
import pandas as pd

class Die:
    '''
    May be rolled to select a face.
    Has N sides.
    Each side is associated with a face and a weight.
    Weights default to 1.0 and can be changed.

    Class variables:
        _roll_is_being_tested: bool -- an indicator of whether roll is being tested

    Instance variables:
        _data_frame_of_faces_and_weights: pd.DataFrame -- a data frame with an index of faces and a column of weights. A face has a data type of str, int, or float. All faces in the index have the same data type. The faces must be unique. A weight has a data type of float.

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
            ValueError, if the provided face is not in the index of this Die object's dataframe of faces and weights, or if the provided weight cannot be converted to a np.float64 object

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
            number_of_rolls: int -- An integer

        Return values:
            list_of_rolled_faces: list -- A list of rolled faces

        Side effects:
            Samples rows from the data frame of faces and weights of this Die object according to the weights
        
        Exceptions raised:
            none

        Restrictions on when this method can be called:
            none
        '''
        
        the_random_state = 0 if Die._roll_is_being_tested else None
        if self._data_frame_of_faces_and_weights.shape[0] > 0:
            type_of_face = type(self._data_frame_of_faces_and_weights.at[0, 'face'])
        else:
            type_of_face = None
        data_frame_of_rolled_faces_and_weights = self._data_frame_of_faces_and_weights.sample(n = number_of_rolls, replace = True, weights = 'weight', random_state = the_random_state, axis = None, ignore_index = False).reset_index(drop = True)
        list_of_rolled_faces = [type_of_face(element) for element in data_frame_of_rolled_faces_and_weights['face'].to_list()]
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

class Game:
    '''
    Encapsulates a list of one or more dice with the same set of faces and methods to play by rolling one or more times all dice in the list and show a data frame of rolls and dice or a data frame of rolls, dice, and faces

    Instance variables:
        _data_frame_of_rolls_and_dice: pd.DataFrame -- a data frame of rolls and dice, where the number of rows and observations is the number of rolls, the number of columns and features is the number of dice, and each cell value is a face rolled

    Public methods:
        __init__
        play
        show
    '''

    def __init__(self, list_of_dice):
        '''
        Initializes a Game object with a list of one or more dice with the same set of faces
        
        Keyword arguments:
            list_of_dice: list -- a list of one or more dice with the same set of faces

        Return values:
            none

        Side effects:
            Initializes this Game object's list of one or more dice with the same set of faces

        Exceptions raised:
            none

        Restrictions on when this method can be called:
            May not be called directly
        '''

        self._list_of_dice = list_of_dice
        self._this_game_has_been_played = False

    def play(self, number_of_rolls):
        '''
        Plays by rolling one or more times all dice in this Game object's list of one or more dice with the same set of faces

        Keyword arguments:
            number_of_rolls: int -- An integer

        Return values:
            none

        Side effects:
            Creates a data frame of rolls and dice, where the number of rows and observations is the number of rolls, the number of columns and features is the number of dice, and each cell value is a face rolled
        
        Exceptions raised:
            none

        Restrictions on when this method can be called:
            none
        '''

        self._data_frame_of_rolls_and_dice = pd.DataFrame()
        self._data_frame_of_rolls_and_dice.index.rename('roll_index', inplace = True)
        for i in range(0, len(self._list_of_dice)):
            die = self._list_of_dice[i]
            self._data_frame_of_rolls_and_dice[i] = die.roll(number_of_rolls)
        self._this_game_has_been_played = True

    def show(self, form):
        '''
        Displays and provides the data frame of rolls and dice of this Game object

        Keyword arguments:
            form: str -- narrow or wide

        Return values:
            the data frame of rolls and dice of this Game object or a version of that data frame in narrow form

        Side effects:
            Displays the data frame of rolls and dice of this Game object or a version of that data frame in narrow form

        Exceptions raised:
            AssertionError if this game has not been played
            ValueError if the provided form is neither narrow nor wide

        Restrictions on when this method can be called:
            none
        '''

        if not self._this_game_has_been_played:
            raise AssertionError('this game has not been played')
        if form == 'narrow':
            data_frame_of_rolls_dice_and_faces = self._data_frame_of_rolls_and_dice.stack().to_frame('face')
            data_frame_of_rolls_dice_and_faces.index.rename(['roll_index', 'die_index'], inplace = True)
            #print(data_frame_of_rolls_dice_and_faces)
            return data_frame_of_rolls_dice_and_faces
        elif form == 'wide':
            #print(self._data_frame_of_rolls_and_dice)
            return self._data_frame_of_rolls_and_dice
        else:
            raise ValueError('the form of the data frame of rolls and dice must be either narrow or wide')

class Analyzer:
    '''
    Encapsulates structures of descriptive statistics for a game that has been played and methods to generate these structures of descriptive statistics

    Instance variables:
        _data_frame_of_rolls_and_face_counts: pd.DataFrame -- a data frame of rolls and face counts, where the number of rows and observations is the number of rolls, the number of columns and features is the number of faces, and each cell value is a count of the number of dice for one roll with a face

    Public methods:
        __init__
        generate_data_frame_of_rolls_and_face_counts
        get_number_of_rolls_where_all_dice_have_the_same_face
        generate_data_frame_of_face_combinations_and_counts
        play
    '''

    def __init__(self, game):
        '''
        Initializes an Analyzer object with a Game object, and
        infers the data type of each face of each die in the Game object's list of dice
        
        Keyword arguments:
            game: Game -- a Game object

        Return values:
            none

        Side effects:
            Initializes this Analyzer object's Game object

        Exceptions raised:
            none

        Restrictions on when this method can be called:
            May not be called directly
        '''

        self._game = game
        data_frame_of_rolls_and_dice = self._game.show('wide')
        face = data_frame_of_rolls_and_dice.at[0, 0]
        self._type_of_face = type(face)
        self._data_frame_of_face_combinations_and_counts_needs_to_be_generated = True

    def generate_data_frame_of_rolls_and_face_counts(self):
        '''
        Generates a data frame of rolls and face counts, where the number of rows and observations is the number of rolls, the number of columns and features is the number of faces, and each cell value is a count of the number of dice for one roll with a face

        Keyword arguments:
            none

        Return values:
            a data frame of rolls and face counts, where the number of rows and observations is the number of rolls, the number of columns and features is the number of faces, and each cell value is a count of the number of dice for one roll with a face

        Side effects:
            Stores a data frame of rolls and face counts, where the number of rows and observations is the number of rolls, the number of columns and features is the number of faces, and each cell value is a count of the number of dice for one roll with a face

        Exceptions raised:
            none

        Restrictions are when this method can be called:
            none
        '''

        data_frame_of_rolls_and_dice = self._game.show('wide')
        self.data_frame_of_rolls_and_face_counts = data_frame_of_rolls_and_dice.apply(lambda series_of_faces: series_of_faces.value_counts(), axis = 1).fillna(0).astype(dtype = self._type_of_face).rename_axis(columns = 'face')
        return self.data_frame_of_rolls_and_face_counts

    def get_number_of_rolls_where_all_dice_have_the_same_face(self, data_frame_of_face_combinations_and_counts_where_combinations_have_all_faces_the_same_should_be_created = True):
        '''
        Gets the number of rolls where all dice for one roll have the same face

        Keyword arguments:
            none

        Return values:
            a number of rolls where all dice have the same face

        Side effects:
            May call generate_data_frame_of_face_combinations_and_counts
            Creates a data frame of face combinations and counts where combinations have all faces the same

        Exceptions raised:
            none

        Restrictions on when this method can be called:
            none
        '''

        if self._data_frame_of_face_combinations_and_counts_needs_to_be_generated:
            self.generate_data_frame_of_face_combinations_and_counts()
            self._data_frame_of_face_combinations_and_counts_needs_to_be_generated = False
        if (data_frame_of_face_combinations_and_counts_where_combinations_have_all_faces_the_same_should_be_created):
            list_with_elements_face = ['face'] * len(self.data_frame_of_face_combinations_and_counts.index[0])
            empty_multiIndex = pd.MultiIndex.from_tuples([], names = list_with_elements_face)
            self.data_frame_of_face_combinations_and_counts_where_combinations_have_all_faces_the_same = pd.DataFrame(index = empty_multiIndex, columns = ['count'])
        number_of_rolls_where_all_dice_have_the_same_face = 0
        for face_combination, series_of_face_combination_and_count in self.data_frame_of_face_combinations_and_counts.iterrows():
            if len(set(face_combination)) == 1:
                number_of_rolls_where_all_dice_have_the_same_face += series_of_face_combination_and_count['count']
                if (data_frame_of_face_combinations_and_counts_where_combinations_have_all_faces_the_same_should_be_created):
                    self.data_frame_of_face_combinations_and_counts_where_combinations_have_all_faces_the_same.loc[face_combination, :] = series_of_face_combination_and_count['count']
        return number_of_rolls_where_all_dice_have_the_same_face

    def generate_data_frame_of_face_combinations_and_counts(self):
        '''
        Generates a data frame of face combinations and counts of how many times each face combination was rolled

        Keyword arguments:
            none

        Return values:
            a data frame of face combinations and counts of how many times each face combination was rolled

        Side effects:
            Stores a data frame of face combinations and counts of how many times each face combination was rolled

        Exceptions raised:
            none

        Restrictions on when this method can be called:
            none
        '''

        data_frame_of_rolls_and_dice = self._game.show('wide')
        number_of_faces = data_frame_of_rolls_and_dice.shape[1]
        list_with_elements_face = ['face'] * number_of_faces
        empty_multiIndex = pd.MultiIndex.from_tuples([], names = list_with_elements_face)
        self.data_frame_of_face_combinations_and_counts = pd.DataFrame(index = empty_multiIndex, columns = ['count'])
        for roll_index, series_of_faces in data_frame_of_rolls_and_dice.iterrows():
            list_of_faces = series_of_faces.to_list()
            list_of_sorted_faces = sorted(list_of_faces)
            face_combination = tuple(list_of_sorted_faces)
            if self.data_frame_of_face_combinations_and_counts.index.isin([face_combination]).any():
                self.data_frame_of_face_combinations_and_counts.at[face_combination, 'count'] += 1
            else:
                self.data_frame_of_face_combinations_and_counts.at[face_combination, 'count'] = 1
        self._data_frame_of_face_combinations_and_counts_needs_to_be_generated = False
        return self.data_frame_of_face_combinations_and_counts

    def play(self, number_of_rolls):
        '''
        Plays this analyzer's game and indicates that this analyzer's data frame of face combinations and counts needs to be generated

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Plays this analyzer's game and indicates that this analyzer's data frame of face combinations and counts needs to be generated

        Exceptions raised:
            none

        Restrictions on when this method can be called:
            none
        '''

        self._game.play(number_of_rolls)
        self._data_frame_of_face_combinations_and_counts_needs_to_be_generated = True