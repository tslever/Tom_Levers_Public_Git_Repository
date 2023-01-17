from booklover.booklover import BookLover
book_lover = BookLover('Han Solo', 'hsolo@millenniumfalcon.com', 'scifi')
book_lover.add_book('Star Wars: A New Hope', 5)
print(book_lover.num_books_read())