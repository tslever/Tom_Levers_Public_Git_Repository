'''
Module for class BookLoverTestSuite, which tests the methods of a BookLover
'''

import unittest
from booklover.booklover import *

class BookLoverTestSuite(unittest.TestCase):
    '''
    Tests the methods of a BookLover

    Instance variables:
        none

    Public methods:
        test_1_init
    '''

    def setUp(self):
        '''
        Creates a instance variable with a value of a BookLover

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Creates an instance variable with a value of a BookLover

        Exceptions raised:
            none

        Restrictions on when this method can be called:
            none
        '''

        self.book_lover = BookLover('Han Solo', 'hsolo@millenniumfalcon.com', 'scifi')

    def test_0_init(self):
        '''
        Tests BookLover.__init__

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Tests whether a BookLover's name, email, favorite genre, number of read books, and book list are equal to expected values

        Exceptions raised:
            AssertionError if a BookLover's name, email, favorite genre, number of read books, or book list is not equal to an expected value

        Restrictions on when this method can be called:
            none
        '''

        self.assertEqual(self.book_lover.name, 'Han Solo')
        self.assertEqual(self.book_lover.email, 'hsolo@millenniumfalcon.com')
        self.assertEqual(self.book_lover.fav_genre, 'scifi')
        self.assertEqual(self.book_lover.num_books, 0)
        self.assertTrue(self.book_lover.book_list.equals(pd.DataFrame({'book_name':[], 'book_rating':[]}).astype(dtype = {'book_name': 'str', 'book_rating': 'int'})))


    def test_1_add_book(self):
        '''
        Tests BookLover.add_book by confirming that a book was successfully added to a BookLover's read book list

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Adds a book to the read-book list of a BookLover and tests whether the book is in the BookLover's read-book list

        Exceptions raised:
            AssertionError if a book, including book name and rating, is not correctly added to the read-book list of a BookLover

        Restrictions on when this method can be called:
            none
        '''
        
        self.book_lover.add_book('Star Wars: A New Hope', 5)
        self.assertTrue(self.book_lover.book_list.equals(pd.DataFrame({'book_name': ['Star Wars: A New Hope'], 'book_rating': [5]})))

    def test_2_add_book(self):
        '''
        Tests BookLover.add_book by running test_1_add_book twice, catching a BookAlreadyExistsInBookListException thrown when an attempt is made to add a book already in a BookLover's read-book list, and confirming that only one book was added to the BookLover's read-book list

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Runs test_1_add_book twice, catches a BookAlreadyExistsInBookListException thrown when an attempt is made to add a book already in a BookLover's read book list, and confirms that only one book was added to the BookLover's read-book list

        Exceptions raised:
            AssertionError if test_1_add_book succeeds a second time or a BookLover's read-book list does not contain one book ultimately

        Restrictions on when this method can be called:
            none
        '''
        
        self.test_1_add_book()
        try:
            self.test_1_add_book()
            self.fail()
        except BookAlreadyExistsInBookListException as e:
            pass
        self.assertTrue(self.book_lover.book_list.equals(pd.DataFrame({'book_name': ['Star Wars: A New Hope'], 'book_rating': [5]})))

    def test_3_has_read(self):
        '''
        Tests BookLover.has_read by ensuring that has_read returns True when the name of a book in a BookLover's read-book list is provided to has_read

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Ensures that has_read returns True when the name of a book in a BookLover's read-book list is provided to has_read

        Exceptions raised:
            AssertionError if has_read returns False when the name of a book in a BookLover's read-book list is provided to has_read

        Restrictions on when this method can be called:
            none
        '''

        self.test_1_add_book()
        self.assertTrue(self.book_lover.has_read('Star Wars: A New Hope'))

    def test_4_has_read(self):
        '''
        Tests BookLover.has_read by ensuring that has_read returns False when the name of a book not in a BookLover's read-book list is provided to has_read

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Ensures that has_read returns False when the name of a book not in a BookLover's read-book list is provided to has_read

        Exceptions raised:
            AssertionError if has_read returns True when the name of a book not in a BookLover's read-book list is provided to has_read

        Restrictions on when this method can be called:
            none
        '''

        self.test_1_add_book()
        self.assertFalse(self.book_lover.has_read('Star Wars: Empire Strikes Back'))

    def test_5_num_books_read(self):
        '''
        Tests BookLover.num_books_read by ensuring that num_books_read returns 2 when a BookLover's read-book list contains two books

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Ensures that num_books_read returns 2 when a BookLover's read-book list contains two books

        Exceptions raised:
            AssertionError if num_books_read returns a number other than 2 when a BookLover's read-book list contains two books

        Restrictions on when this method can be called:
            none
        '''

        self.test_1_add_book()
        self.book_lover.add_book('Star Wars: Empire Strikes Back', 4)
        self.assertEqual(self.book_lover.num_books_read(), 2)

    def test_6_fav_books(self):
        '''
        Tests BookLover.fav_books by ensuring that fav_books returns a data frame of two books with ratings greater than 3 when a BookLover's read-book list contains two books with rating greater than 3 and one book with rating 3

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Ensures that fav_books returns a data frame of two books when a BookLover's read-book list contains two books with rating greater than 3 and one book with rating 3

        Exceptions raised:
            AssertionError if fav_book does not return a data frame of two books with ratings greater than 3
            when a BookLover's read-book list contains two books with rating greater than 3 and one book with rating 3

        Restrictions on when this method can be called:
            none
        '''

        self.test_1_add_book()
        self.book_lover.add_book('Star Wars: Empire Strikes Back', 4)
        self.book_lover.add_book('Star Wars: Return of the Jedi', 3)
        data_frame_of_favorite_books = self.book_lover.fav_books()
        self.assertTrue(
            data_frame_of_favorite_books.equals(
                pd.DataFrame(
                    {
                        'book_name': [
                            'Star Wars: A New Hope',
                            'Star Wars: Empire Strikes Back'
                        ],
                        'book_rating': [
                            5,
                            4
                        ]
                    }
                )
            )
        )
        series_of_indicators_of_whether_ratings_are_greater_than_3 = (data_frame_of_favorite_books['book_rating'] > 3)
        self.assertTrue(series_of_indicators_of_whether_ratings_are_greater_than_3.all())

    def tearDown(self):
        '''
        Deletes an instance variable with a value of a BookLover

        Keyword arguments:
            none

        Return values:
            none

        Side effects:
            Deletes an instance variable with a value of a BookLover

        Exceptions raised:
            none

        Restrictions on when this method can be called:
            none
        '''

        del self.book_lover

if __name__ == '__main__':
    verbose = 3
    unittest.main(verbosity = verbose)