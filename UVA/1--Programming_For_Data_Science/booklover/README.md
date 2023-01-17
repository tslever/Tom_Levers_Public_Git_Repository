# booklover
A Python package containing class BookLover


Installation

From the root of this package, run `pip install .`.


Usage

In a Python interactive shell, while I can run `import booklover`, it is not useful.
`b = BookLover('name', 'email', 'fav_genre')` throws NameError: name 'BookLover' is not defined.
`b = booklover.BookLover('name', 'email', 'fav_genre')` throws AttributeError: module 'booklover' has no attribute 'BookLover'.
`b = booklover.booklover.BookLover('name', 'email', 'fav_genre')` throws AttributeError: module 'booklover' has no attribute 'booklover'.

I can run `import booklover.booklover`.
`b = booklover.booklover.BookLover('name', 'email', 'fav_genre')` succeeds.

I can run `from booklover import booklover`.
`b = booklover.BookLover('name', 'email', 'fav_genre')` succeeds.

I can run `from booklover.booklover import BookLover`.
`b = BookLover('name', 'email', 'fav_genre')` succeeds.

I can run `!python test_modules/booklover_test.py`.