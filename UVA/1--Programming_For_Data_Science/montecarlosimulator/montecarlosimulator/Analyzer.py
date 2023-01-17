'''
Module for class Analyzer, which generates structures of descriptive statistics for a game that has been played
'''

import pandas as pd

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