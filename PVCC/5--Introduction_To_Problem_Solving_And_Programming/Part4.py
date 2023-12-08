# Tom Lever
# 12/08/2023
# Introduction to Problem Solving and Programming
# Part4.py allows a user to encode plaintext or decode cyphertext

# To test, test
# - Decoding cyphertext loaded from a nonexisting file
# - Decoding cyphertext loaded from an existing file
# - Encoding plaintext into cyphertext

import os

plaintext_or_cyphertext = input('Do you wish to enter plaintext to convert to cyphertext or decode existing cyphertext (plaintext / cyphertext)? ')
if plaintext_or_cyphertext == 'cyphertext':
    if os.path.exists('cypher.txt'):
        with open('cypher.txt', 'r') as file:
            cyphertext = file.read()
            print(f'I read cyphertext {cyphertext}.')
            distance = input('Enter integer distance: ')
            print(f'You entered distance {distance}.')
            # TODO: Decode cyphertext into plaintext
            plaintext = 'placeholder plaintext'
            print(f'Your cyphertext is decoded into plaintext "{plaintext}".')
    else:
        raise FileExistsError('cypher.txt does not exist: Use this program to convert plaintext into cyphertext.')
elif plaintext_or_cyphertext == 'plaintext':
    plaintext = input('Enter any printable character plaintext: ')
    print(f'You entered plaintext "{plaintext}".')
    distance = input('Enter integer distance: ')
    print(f'You entered distance {distance}.')
    # TODO: Encode plaintext into cyphertext
    cyphertext = 'placeholder cyphertext'
    print(f'Your plaintext is encoded into cyphertext "{cyphertext}".')
    with open('cypher.txt', 'w') as file:
        file.write(cyphertext)
    print('Your cyphertext is written to cypher.txt.')
else:
    print('Your entry is not supported.')
input()