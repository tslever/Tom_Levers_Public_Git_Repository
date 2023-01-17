# montecarlosimulator

A Python package offering a Monte-Carlo simulator

Created: 10/01/22 by Tom Lever

Updated: 12/01/22 by Tom Lever

## Metadata

Author: Tom Lever

Project name: Monte Carlo Simulator

## Synopsis

### Creating montecarlosimulator

Regarding creating this package: https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/

PEP 8 - Style Guide for Python Code:  https://peps.python.org/pep-0008/#package-and-module-names

PEP 257 - Docstring Conventions: https://peps.python.org/pep-0257/

### Installing

To install from root of `montecarlosimulator`: `pip install .`

To install from GitHub: `pip install git+ssh://git@github.com/tslever/montecarlosimulator.git@main`

## Importing

To import class `Die`, run `from montecarlosimulator import Die` or `from montecarlosimulator.montecarlo import Die`.

To import class `Game`, run `from montecarlosimulator import Game`.

To import class `Analyzer`, run `from montecarlosimulator import Analyzer`.

## Creating dice

Run the following code to
* create a die,
* show the die's data frame of faces and weights,
* change a weight of a face of the die,
* show the die's data frame of faces and weights,
* roll the die and acquire a list of rolled faces, and
* print the list of rolled faces.

        import numpy as np
        from montecarlosimulator import Die
        array_of_faces = np.array(['A', 'B', 'C'], dtype = str)
        die = Die(array_of_faces)
        print(die.show())

          face  weight
        0    A     1.0
        1    B     1.0
        2    C     1.0

        die.change_weight(face = 'A', weight = 2.0)
        print(die.show())

          face  weight
        0    A     2.0
        1    B     1.0
        2    C     1.0

        list_of_rolled_faces = die.roll(number_of_rolls = 3)
        print(list_of_rolled_faces)

        ['B', 'C', 'A']

## Playing games

Run the following code after creating one or more dice to

* create a second die,

* show the second die's data frame of faces and weights

* create a list of dice,

* create a game,

* play the game by rolling the dice $1000$ times,

* show the game's data frame of rolls and dice in wide form, and

* show a version of the game's data frame of rolls and dice in narrow form

        die_2 = Die(array_of_faces)
        print(die_2.show())

          face  weight
        0    A     1.0
        1    B     1.0
        2    C     1.0

        list_of_dice = [die, die_2]
        from montecarlosimulator import Game
        game = Game(list_of_dice)
        game.play(number_of_rolls = 1000)
        print(game.show(form = 'wide'))

                    0  1
        roll_index
        0           A  C
        1           C  C
        2           A  A
        3           B  C
        4           B  A
        ...        .. ..
        995         B  C
        996         B  B
        997         C  C
        998         B  A
        999         B  A

        print(game.show(form = 'narrow'))

                             face
        roll_index die_index
        0          0            A
                   1            C
        1          0            C
                   1            C
        2          0            A
        ...                   ...
        997        1            C
        998        0            B
                   1            A
        999        0            B
                   1            A

## Analyzing games

Run the following code after playing a game to

* reate an analyzer,

* generate and print a data frame of rolls and face counts,

* get and print a number of rolls where all dice have the same face,

* generate and print a data frame of face combinations and counts,

* replay the game of the analyzer, and

* regenerate and print a data frame of rolls and face counts.

        from montecarlosimulator import Analyzer
        analyzer = Analyzer(game)
        data_frame_of_rolls_and_face_counts = analyzer.generate_data_frame_of_rolls_and_face_counts()
        print(data_frame_of_rolls_and_face_counts)

        face          A    B    C
        roll_index
        0           0.0  0.0  2.0
        1           1.0  0.0  1.0
        2           0.0  2.0  0.0
        3           1.0  0.0  1.0
        4           1.0  1.0  0.0
        ...         ...  ...  ...
        995         1.0  1.0  0.0
        996         0.0  1.0  1.0
        997         0.0  0.0  2.0
        998         1.0  0.0  1.0
        999         2.0  0.0  0.0

        number_of_rolls_where_all_dice_have_the_same_face = analyzer.get_number_of_rolls_where_all_dice_have_the_same_face()
        print(number_of_rolls_where_all_dice_have_the_same_face)

        300

        data_frame_of_face_combinations_and_counts = analyzer.generate_data_frame_of_face_combinations_and_counts()
        print(data_frame_of_face_combinations_and_counts)

                  count
        face face
        C    C       59
        A    C      276
        B    B       75
        A    B      247
        B    C      177
        A    A      166

        analyzer.play(1000)
        data_frame_of_rolls_and_face_counts = analyzer.generate_data_frame_of_rolls_and_face_counts()
        print(data_frame_of_rolls_and_face_counts)

        face          A    B    C
        roll_index
        0           0.0  2.0  0.0
        1           1.0  0.0  1.0
        2           1.0  1.0  0.0
        3           2.0  0.0  0.0
        4           1.0  1.0  0.0
        ...         ...  ...  ...
        995         2.0  0.0  0.0
        996         0.0  0.0  2.0
        997         1.0  1.0  0.0
        998         2.0  0.0  0.0
        999         0.0  1.0  1.0

## API description

### Die

#### Description

A die may be rolled to select a face.
A die has $N$ sides.
Each side is associated with a face and a weight.
Weights default to $1.0$ and may be changed.
A face has a data type of `str`, `int`, or `float`.
All faces have the same data type.
The faces must be unique.
A weight has a data type of `float`.

#### Public methods

`__init__`

`change_weight`

`roll`

`show`

##### __init__

###### Docstring

Initializes a Die object

Keyword arguments:

`array_of_faces`: `np.ndarray` -- a 1D `numpy` array of faces. A face must have a data type of `str`, `int`, or `float`. All faces in the `numpy` array have the same data type. The faces in the `numpy` array must be unique.

Return values:

none

Side effects:

Initializes this Die object's data frame of faces and weights

Exceptions raised:

none

Restrictions on when this method can be called:

May not be called directly

###### Keyword arguments

`array_of_faces`: `np.ndarray` -- a 1D `numpy` array of faces. A face must have a data type of `str`, `int`, or `float`. All faces in the `numpy` array have the same data type. The faces in the `numpy` array must be unique.

###### Return values

none

##### change_weight

###### Docstring

Changes the weight of a provided face to a provided weight

Keyword arguments:

`face`: `str`, `int`, `np.float64` -- A string, integer, or floating-point number

`weight`: `np.float64` -- A numpy floating-point number

Return values:

none

Side effects:

Changes the weight of the provided face to the provided weight

Exceptions raised:

`ValueError`, if the provided face is not in the index of this `Die` object's data frame of faces and weights, or if the provided weight cannot be converted to a `np.float64` object

Restrictions on when this method can be called:

none

###### Keyword arguments

`face`: `str`, `int`, `np.float64` -- A string, integer, or floating-point number

`weight`: `np.float64` -- A `numpy` floating-point number

###### Return values

none

##### roll

###### Docstring

Rolls this `Die` object one or more times

Keyword arguments:

`number_of_rolls`: `int` -- An integer. Defaults to $1$.

Return values:

`list_of_rolled_faces`: `list` -- A list of rolled faces

Side effects:

Samples rows from the data frame of faces and weights of this `Die` object according to the weights

Exceptions raised:

none

Restrictions on when this method can be called:

none

###### Keyword arguments

`number_of_rolls`: `int` -- An integer. Defaults to $1$.

###### Return values

`list_of_rolled_faces`: `list` -- A list of rolled faces

##### show

###### Docstring

Displays and provides the data frame of faces and weights of this `Die` object

Keyword arguments:

none

Return values:

`_data_frame_of_faces_and_weights`: `pd.DataFrame` -- The data frame of faces and weights of this `Die` object

Side effects:

Displays the data frame of faces and weights of this `Die` object

Exceptions raised:

none

Restrictions on when this method can be called:

none

###### Keyword arguments

none

###### Return values

`_data_frame_of_faces_and_weights`: `pd.DataFrame` -- The data frame of faces and weights of this `Die` object

### Game

#### Description

Encapsulates a list of one or more dice with the same set of faces, a method to play this game by rolling one or more times all dice in the list, and a method to show a data frame of rolls and dice or a data frame of rolls, dice, and faces

#### Public methods

`__init__`

`play`

`show`

##### __init__

###### Docstring

Initializes a `Game` object with a list of one or more dice with the same set of faces

Keyword arguments:

`list_of_dice`: `list` -- a list of one or more dice with the same set of faces

Return values:

none

Side effects:

Initializes this `Game` object's list of one or more dice with the same set of faces

Exceptions raised:

none

Restrictions on when this method can be called:

May not be called directly

###### Keyword arguments

`list_of_dice`: `list` -- a list of one or more dice with the same set of faces

###### Return values

none

##### play

###### Docstring

Plays by rolling one or more times all dice in this `Game` object's list of one or more dice

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

###### Keyword arguments

`number_of_rolls`: `int` -- An integer

###### Return values

none

##### show

###### Docstring

Displays and provides the data frame of rolls and dice of this `Game` object

Keyword arguments:

`form`: `str` -- narrow or wide

Return values:

the data frame of rolls and dice of this Game object or a version of that data frame in narrow form

Side effects:

Displays the data frame of rolls and dice of this `Game` object or a version of that data frame in narrow form

Exceptions raised:

`AssertionError` if this game has not been played

`ValueError` if the provided form is neither narrow nor wide

Restrictions on when this method can be called:

none

###### Keyword arguments

`form`: `str` -- narrow or wide

###### Return values

none

### Analyzer

#### Description

Encapsulates structures of descriptive statistics for a game that has been played and methods to generate these structures of descriptive statistics

#### Public methods

`__init__`

`generate_data_frame_of_rolls_and_face_counts`

`get_number_of_rolls_where_all_dice_have_the_same_face`

`generate_data_frame_of_face_combinations_and_counts`

`play`

##### __init__

###### Docstring

Initializes an `Analyzer` object with a `Game` object, and
infers the data type of each face of each die in the `Game` object's list of dice

Keyword arguments:

`game`: `Game` -- a `Game` object

Return values:

none

Side effects:

Initializes this `Analyzer` object's `Game` object

Exceptions raised:

none

Restrictions on when this method can be called:

May not be called directly

###### Keyword arguments

`game`: `Game` -- a `Game` object

###### Return values

none

##### generate_data_frame_of_rolls_and_face_counts

###### Docstring

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

###### Keyword arguments

none

###### Return values

a data frame of rolls and face counts, where the number of rows and observations is the number of rolls, the number of columns and features is the number of faces, and each cell value is a count of the number of dice for one roll with a face

##### get_number_of_rolls_where_all_dice_have_the_same_face

###### Docstring

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

###### Keyword arguments

none

###### Return values

a number of rolls where all dice have the same face

##### generate_data_frame_of_face_combinations_and_counts

###### Docstring

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

###### Keyword arguments

none

###### Return values

a data frame of face combinations and counts of how many times each face combination was rolled

##### play

###### Docstring

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

###### Keyword arguments

none

###### Return values

none