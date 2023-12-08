# Tom Lever
# 12/08/2023
# Introduction to Problem Solving and Programming
# Part4.py allows a user to encode plaintext or decode cyphertext

# To test, test
# - Decoding cyphertext loaded from a nonexisting file
# - Decoding cyphertext loaded from an existing file
# - Encoding plaintext into cyphertext

import os

# Source: ChatGPT 3.5
def caesar_cipher(text, shift):
    result = ""
    for char in text:
        if char.isalpha():
            is_upper = char.isupper()
            char = char.lower()
            shifted_char = chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
            if is_upper:
                shifted_char = shifted_char.upper()
            result += shifted_char
        else:
            result += char
    return result

plaintext_or_cyphertext = input('Do you wish to enter plaintext to convert to cyphertext or decode existing cyphertext (plaintext / cyphertext)? ')
if plaintext_or_cyphertext == 'cyphertext':
    if os.path.exists('cypher.txt'):
        with open('cypher.txt', 'r') as file:
            cyphertext = file.read()
            print(f'I read cyphertext "{cyphertext}".')
            string_representing_distance = input('Enter integer distance: ')
            distance = int(string_representing_distance)
            print(f'You entered distance {distance}.')
            plaintext = caesar_cipher(cyphertext, -distance)
            print(f'Your cyphertext is decoded into plaintext "{plaintext}".')
    else:
        print('cypher.txt does not exist: Use this program to convert plaintext into cyphertext.')
elif plaintext_or_cyphertext == 'plaintext':
    plaintext = input('Enter any printable character plaintext: ')
    print(f'You entered plaintext "{plaintext}".')
    string_representing_distance = input('Enter integer distance: ')
    distance = int(string_representing_distance)
    print(f'You entered distance {distance}.')
    cyphertext = caesar_cipher(plaintext, distance)
    print(f'Your plaintext is encoded into cyphertext "{cyphertext}".')
    with open('cypher.txt', 'w') as file:
        file.write(cyphertext)
    print('Your cyphertext is written to cypher.txt.')
else:
    print('Your entry is not supported.')
input()