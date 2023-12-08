# Tom Lever
# 12/08/2023
# Introduction to Problem Solving and Programming
# Part4.py allows a user to encode plaintext or decode cyphertext

# Test trying to decode cyphertext in a nonexistent file.
# Test decoding.
# Test encoding.

plaintext_or_cyphertext = input('Do you wish to enter plaintext to convert to cyphertext or decode existing cyphertext (plaintext / cyphertext)? ')
if plaintext_or_cyphertext == 'cyphertext':
    # If file cypher.txt exists,
    #     Ask for integer distance
    #     Decode
    #     Show plaintext
    pass
elif plaintext_or_cyphertext == 'plaintext':
    # Ask user to enter integer distance
    # Let user know what distance user entered
    # Ask user for any printable character plaintext
    # Let user know what plaintext user entered
    # Convert plaintext to cyphertext
    # Let user know that plaintext is converted to cyphertext
    print('Your plaintext is encoded.')
    # Save cyphertext in file cypher.txt
    # Let user know that cyphertext is written to cypher.txt
    print("Your cyphertext is written to cypher.txt.")
else:
    print('Your entry is not supported.')
input()