from array import array
import sys


def translate(base, array_of_numbers_as_strings):

    string = ""
    for number_as_string in array_of_numbers_as_strings:
        if base == "10":
            integer = int(number_as_string, 10)
        elif base == "16":
            integer = int("0x" + number_as_string, 16)
        string += chr(integer)
    print(string)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("\nPass interpreter 'python' the name of this script, the number base, and an array of hexadecimal numbers written as '[n1 n2 n3... nn]'.\n")
        print("Include single or double quotes.")
        assert(False)

    base = sys.argv[1]
    array_of_tokens = sys.argv[2].strip("[]").split()
    array_of_numbers_as_strings = [token.strip(",") for token in array_of_tokens]

    translate(base, array_of_numbers_as_strings)