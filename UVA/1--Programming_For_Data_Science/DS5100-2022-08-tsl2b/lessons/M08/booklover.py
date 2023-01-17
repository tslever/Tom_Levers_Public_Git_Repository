'''
Module for class BookLover
'''

from BookAlreadyExistsInBookListException import BookAlreadyExistsInBookListException
import pandas as pd

class BookLover:
    '''
    A BookLover represents a person with a name, an email address, a favorite genre, a number of read-books, and a read-book list
    who can add a read book, return a number of indicator of whether this BookLover has read a book, return a number of read books, and return a list of favorite books.

    Instance variables:
        name: str -- the name of this BookLover
        email: str -- the email and unique identifier of this BookLover
        fav_genre: str -- the favorite genre of this BookLover (e.g., 'mystery', 'fantasy', 'historical fiction')
        num_books: int -- the number of books that this BookLover has read
        book_list: pd.DataFrame --
            a data frame with columns labeled 'book_name' and 'book_rating' containing read books (i.e., titles of books this BookLover has read and
            this BookLover's ratings of those books on a scale of 1 to 5, where 1 means this BookLover did not like the book at all, and 5 means this BookLover loved the book)
            (e.g., pd.DataFrame(columns =
                ['book_name', 'book_rating'], data = [
                ['Jane Eyre', 4],
                ['Fight Club', 3],
                ['The Divine Comedy', 5],
                ['The Popol Vuh', 5]]
            ))

    Public methods:
        __init__
    '''

    def __init__(
        self,
        name,
        email,
        fav_genre,
        num_books = 0,
        book_list = pd.DataFrame({'book_name':[], 'book_rating':[]}).astype(dtype = {'book_name': str, 'book_rating': int})
    ):
        '''
        Initializes a BookLover
        
        Keyword arguments:
            name: str -- a name with which to initialize this BookLover
            email: str --an email address and unique identifier with which to initialize this BookLover
            fav_genre: str -- a favorite genre with which to initialize this BookLover
            num_books: int -- a number of read books with which to initialize this BookLover. Default 0.
            book_list: pd.DataFrame --
                a data frame with which to initialize this BookLover. Has columns labeled 'book_name' and 'book_rating' and contains read books (i.e., titles of books this BookLover
                has read and this BookLover's ratings of those books on a scale of 0 to 5, where 0 means this BookLover did not like the book at all, and 5 means this BookLover loved
                the book). Default pd.DataFrame({'book_name':[], 'book_rating':[]}).astype(dtype = {'book_name': str, 'book_rating': int}).

        Return values:
            none

        Side effects:
            Initializes this BookLover's name, email address, favorite genre, number of read books, and read-book list

        Exceptions raised:
            none

        Restrictions on when this method can be called:
            May not be called directly
        '''

        self.name = name
        self.email = email
        self.fav_genre = fav_genre
        self.num_books = num_books
        # book_list references an empty data frame of the list __defaults__ of this method of BookLover; this data frame is a class attribute. self.book_list references a instance attribute.
        self.book_list = book_list.copy()

    def _get_list_of_book_names(self):
        self.series_of_book_names = self.book_list['book_name']
        self.list_of_book_names = self.series_of_book_names.to_list()
        return self.list_of_book_names

    def add_book(self, book_name, rating):
        '''
        Adds a read book to this BookLover's read-book list
        
        Keyword arguments:
            book_name: str -- the name of a book to add to this BookLover's read-book list
            rating: int -- a book rating from 0 to 5 to add to this BookLover's read-book list

        Return values:
            none

        Side effects:
            Adds a book to this BookLover's read-book list

        Exceptions raised:
            BookAlreadyExistsException if a book already exists in this BookLover's read-book list

        Restrictions on when this method can be called:
            none
        '''

        if (book_name in self._get_list_of_book_names()):
            raise BookAlreadyExistsInBookListException(book_name + ' already exists in the read-book list of ' + self.name + '.')
        else:
            # Raf Alvarado's solution:
            #new_book = pd.DataFrame({
            #    'book_name': [book_name],
            #    'book_rating': [rating]
            #})
            #self.book_list = pd.concat([self.book_list, new_book], ignore_index = True)
            self.book_list.loc[len(self.book_list.index)] = [book_name, rating]
            self.num_books += 1

    def has_read(self, book_name):
        '''
        Returns an indicator of whether this BookLover has read a book with a specified name
        
        Keyword arguments:
            book_name: str -- the name of a book for which to return an indicator of whether this BookLover has read that book

        Return values:
            True if this BookLover has read a book with a specified name
            False if this BookLover has not read a book with a specified name

        Side effects:
            none

        Exceptions raised:
            none

        Restrictions on when this method can be called:
            none
        '''

        if (book_name in self._get_list_of_book_names()):
            return True
        else:
            return False

    def num_books_read(self):
        '''
        Returns the number of books this BookLover has read
        
        Keyword arguments:
            none

        Return values:
            the total number of books this BookLover has read

        Side effects:
            none

        Exceptions raised:
            none

        Restrictions on when this method can be called:
            none
        '''

        return self.num_books

    def fav_books(self):
        '''
        Returns a data frame of books to which this BookLover has given ratings greater than 3
        
        Keyword arguments:
            none

        Return values:
            a data frame of books to which this BookLover has given ratings greater than 3

        Side effects:
            none

        Exceptions raised:
            none

        Restrictions on when this method can be called:
            none
        '''

        return self.book_list[self.book_list['book_rating'] > 3]