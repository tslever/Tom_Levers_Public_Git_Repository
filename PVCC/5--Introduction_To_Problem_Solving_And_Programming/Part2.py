'''
Tom Lever
11/16/2023
Introduction To Problem Solving And Programming
Compares with 50 the product of two inputted numbers
'''

import numpy as np
first_number = np.float64(input('first number: '))
second_number = np.float64(input('second number: '))
product = first_number * second_number
if product < 50:
    print(f'The numbers you gave were {first_number} and {second_number}, and their product is {product}, which is less than 50')
else:
    print(f'The numbers you gave were {first_number} and {second_number}, and their product is {product}, which is greater than or equal to 50')
input()