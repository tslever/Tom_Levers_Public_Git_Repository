
import numpy as np
import sys


def put_matrix_in_reduced_row_echelon_form(A):

    print()
    print("Initial matrix:")
    print(A)
    print()

    print("Shape of initial matrix:")
    (m, n) = A.shape
    print("(number of rows m, number of columns n) = " + str(A.shape))
    print()

    print("Initializing iteration number to 0.")
    print("Initializing row index i to 0.")
    print("Initializing column index j to 0.")
    i = 0
    j = 0
    iteration_number = 0
    print()

    print(f"Entering a loop while [i < (m={m})] and [j < (n={n})].")
    print()
    while (i < m) and (j < n):

        print(f"Iteration {iteration_number}.")
        print(f"Row index i = {i}.")
        print(f"Column index j = {j}.")
        print()

        print(f"Column (j={j}):")
        column_j = A[:, j]
        print(column_j)
        print()

        print(f"Column (j={j}) from row index (i={i}) to row index [(m={m}) - 1] inclusive:")
        column_j_from_i_to_m_minus_1 = A[i:m, j]
        print(column_j_from_i_to_m_minus_1)
        print()

        print(f"Magnitudes of elements in column (j={j}) from row index (i={i}) to row index [(m={m}) - 1] inclusive:")
        magnitudes_in_column_j_from_i_to_m_minus_1 = abs(column_j_from_i_to_m_minus_1)
        print(magnitudes_in_column_j_from_i_to_m_minus_1)
        print()

        print(f"Maximum magnitude among elements in column (j={j}) from row index (i={i}) to row index [(m={m}) - 1] inclusive:")
        maximum_magnitude_in_column_j_from_i_to_m_minus_1 = max(magnitudes_in_column_j_from_i_to_m_minus_1)
        print(maximum_magnitude_in_column_j_from_i_to_m_minus_1)
        print()

        print(f"Index of maximum magnitude in column (j={j}) from row index (i={i}) to row index [(m={m}) - 1] inclusive:")
        index_of_maximum_magnitude_in_column_j_from_i_to_m_minus_1 = np.argmax(magnitudes_in_column_j_from_i_to_m_minus_1)
        print(index_of_maximum_magnitude_in_column_j_from_i_to_m_minus_1)
        print()

        print(f"Index of maximum magnitude in column (j={j}):")
        index_of_maximum_magnitude_in_column_j = index_of_maximum_magnitude_in_column_j_from_i_to_m_minus_1 + i
        print(index_of_maximum_magnitude_in_column_j)
        print()

        tolerance = 1.6875E-14
        if maximum_magnitude_in_column_j_from_i_to_m_minus_1 <= 0:
            print(f"Maximum magnitude among elements in column (j={j}) from row index (i={i}) to row index [(m={m}) - 1] inclusive is about 0.")
            print(f"Making sure these elements are actually 0.")
            A[i:m, j] = 0
            print(f"Because these elements are all 0, there is no pivot in column (j={j}).")
            print(f"Looking for a pivot in column (j+1 = {j+1}): Incrementing j from {j} to {j+1}.")
            j = j + 1
            print()

        else:

            print(f"Swapping row (i={i}) and and row (<index of maximum magnitude in column j>={index_of_maximum_magnitude_in_column_j}).")
            print("Initial matrix:")
            print(A)
            print("Final matrix:")
            ith_row_initially = A[i, :].copy()
            A[i, :] = A[index_of_maximum_magnitude_in_column_j, :]
            A[index_of_maximum_magnitude_in_column_j, :] = ith_row_initially
            print(A)
            print()

            print(f"Dividing all elements in the pivot row [new row (i={i})] by the pivot {A[i, j]} in the pivot row and pivot column (j={j}).")
            print("Initial matrix:")
            print(A)
            print("Final matrix:")
            A[i, :] = A[i, :] / A[i, j]
            print(A)
            print()
            
            for k in range(0, i):
                print(f"Subtracting, from row {k}, pivot row [row (i={i})] scaled by element {A[k, j]} in row {k} and pivot column (j={j}).")
                print("Initial matrix:")
                print(A)
                print("Final matrix:")
                A[k, :] = A[k, :] - A[k, j] * A[i, :]
                print(A)
                print()

            for k in range(i+1, m):
                print(f"Subtracting, from row {k}, pivot row [row (i={i})] scaled by element {A[k, j]} in row {k} and pivot column (j={j}).")
                print("Initial matrix:")
                print(A)
                print("Final matrix:")
                A[k, :] = A[k, :] - A[k, j] * A[i, :]
                print(A)
                print()

            print(f"Incrementing row index i from {i} to {i+1}.")
            print(f"Incrementing column index j from {j} to {j+1}.")
            print(f"Incrementing iteration number from {iteration_number} to {iteration_number+1}.")
            i = i + 1
            j = j + 1
            iteration_number = iteration_number + 1
            print()

    print(f"Exiting the loop while [i < (m={m})] and [j < (n={n})].")
    print()

    print("Matrix in reduced-row echelon form:")
    print(A)
    print()


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("\nPass interpreter 'python' the name of this script and a matrix written as 'a11, a12, a13; a21, a22, a23'.\n")
        print("Include single quotes in lieu of matrix brackets.")
        assert(False)

    M = np.matrix(sys.argv[1])
    M = M.astype(float)

    put_matrix_in_reduced_row_echelon_form(M)