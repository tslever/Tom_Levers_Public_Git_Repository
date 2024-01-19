package com.tsl.more_operations_with_arrays.input_output_utilities;


import com.tsl.more_operations_with_arrays.App;
import com.tsl.more_operations_with_arrays.AnIntegerOverflowException;
import java.util.InputMismatchException;
import java.util.NoSuchElementException;
import java.util.Scanner;


/**
 * AnInputOutputManager represents the structure for an input output manager with methods to ask about and read
 * one or more numbers.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 */
public class AnInputOutputManager {

	
	/**
	 * scanner is a component of AnInputOutputManager.
	 */
	private Scanner scanner;
	

	/**
	 * askAboutAndReadANumberOfElementsForAnArrayOfRandomIntegers prompts a user to input a number of elements
	 * for an array of random integers and either provides a valid number of elements to the calling method or
	 * prompts the user again.
	 * @return
	 */
	public int askAboutAndReadANumberOfElementsForAnArrayOfRandomIntegers() {
		
		int theProposedNumberOfElements;
		
		while (true) {
			
			System.out.print("Please enter a number of elements for an array of random integers: ");
			
			this.scanner = new Scanner(System.in);
			
			try {
				theProposedNumberOfElements = this.scanner.nextInt();
				
				if (theProposedNumberOfElements < 0) {
					throw new AProposedNumberOfElementsIsNegativeException("The proposed number of elements is negative.");
				}
				
				return theProposedNumberOfElements;
			}
			catch (InputMismatchException theInputMismatchException) {
				System.out.println("Input mismatch exception.");
			}
			catch (NoSuchElementException theNoSuchElementException) {
				System.out.println("No such element exception.");
			}
			catch (AProposedNumberOfElementsIsNegativeException theReadIntegerIsNegativeException) {
				System.out.println(theReadIntegerIsNegativeException.getMessage());
			}
			
		}
		
	}
	
	
	/**
	 * askAboutAndReadALowerLimitAndAnUpperLimitForAnInteger prompts a user to input a lower limit and an upper limit
	 * for an integer and either provides an array with a valid lower limit and a valid upper limit to the calling
	 * method or prompts the user again.
	 * @return
	 */
	public int[] askAboutAndReadALowerLimitAndAnUpperLimitForAnInteger() {
		
		int theProposedLowerLimit;
		int theProposedUpperLimit;
		
		
		while (true) {
		
			try {
				
				while (true) {
					
					System.out.print("Enter the lower limit for an integer: ");
					
					this.scanner = new Scanner(System.in);
					
					try {
						theProposedLowerLimit = this.scanner.nextInt();
						break;
					}
					catch (InputMismatchException theInputMismatchException) {
						System.out.println("Input mismatch exception.");
					}
					catch (NoSuchElementException theNoSuchElementException) {
						System.out.println("No such element exception.");
					}
					
				}
				
				
				while (true) {
					
					System.out.print("Enter the upper limit for an integer: ");
					
					this.scanner = new Scanner(System.in);
					
					try {
						theProposedUpperLimit = this.scanner.nextInt();
						System.out.println();
						break;
					}
					catch (InputMismatchException theInputMismatchException) {
						System.out.println("Input mismatch exception.");
					}
					catch (NoSuchElementException theNoSuchElementException) {
						System.out.println("No such element exception.");
					}
					
				}
				
				
				if (theProposedLowerLimit > theProposedUpperLimit) {
					throw new ALowerLimitIsGreaterThanUpperLimitException(
						"The proposed lower limit is greater than the proposed upper limit.");
				}
				
				return new int[] {theProposedLowerLimit, theProposedUpperLimit};
				
			}
			
			catch (ALowerLimitIsGreaterThanUpperLimitException theLowerLimitIsGreaterThanUpperLimitException) {
				System.out.println(theLowerLimitIsGreaterThanUpperLimitException.getMessage());
			}
		
		}
		
	}
	
}
