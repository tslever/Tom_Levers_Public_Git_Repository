package com.tsl.grading_exams;


import org.junit.jupiter.api.Test;


/**
 * CheckTheAdditionOfTest encapsulates JUnit tests of core functionality of the method checkTheAdditionOf of class
 * AFullArrayListBasedBoundedStackOfRandomIntegers.
 * @author Tom Lever
 * @version 1.0
 * @since 05/29/21
 *
 */
public class CheckTheAdditionOfTest {

	
	/**
	 * THE_MINIMUM_INTEGER is an attribute of checkTheAdditionOfTest.
	 */
	private final int THE_MINIMUM_INTEGER = -2147483648;

	
	/**
	 * THE_MAXIMUM_INTEGER is an attribute of checkTheAdditionOfTest.
	 */
	private final int THE_MAXIMUM_INTEGER = 2147483647;
	
	
	/**
	 * checkTheAdditionOf throws an integer-overflow exception if addition of a first integer and a
	 * second integer would result in a sum greater than the maximum integer or less than the minimum integer.
	 * 
	 * @param theFirstInteger
	 * @param theSecondInteger
	 * @throws AnIntegerOverflowException
	 */
	private void checkTheAdditionOf(int theFirstInteger, int theSecondInteger) throws AnIntegerOverflowException {
		
		if ((theFirstInteger > 0) && (theSecondInteger > 0) && (theFirstInteger > this.THE_MAXIMUM_INTEGER - theSecondInteger) ||
			(theFirstInteger < 0) && (theSecondInteger < 0) && (theFirstInteger < this.THE_MINIMUM_INTEGER - theSecondInteger)) {
			
			throw new AnIntegerOverflowException(
				"Integer-overflow exception: the sum of " + theFirstInteger + " and " + theSecondInteger +
				" is outside the interval [" + this.THE_MINIMUM_INTEGER + ", " + this.THE_MAXIMUM_INTEGER + "]."
			);
			
		}
		
	}
	
	
	/**
	 * testCheckTheAdditionOfForWidestValidRangeOfPositiveIntegers tests
	 * checkTheAdditionOfForWidestValidRangeOfPositiveIntegers by passing a check of the addition of 1 and 2147483646
	 */
	@Test
	public void testCheckTheAdditionOfForWidestValidRangeOfPositiveIntegers() {
		
		System.out.println("Running testCheckTheAdditionOfForWidestValidRangeOfPositiveIntegers.");
		
		try {
			checkTheAdditionOf(1, this.THE_MAXIMUM_INTEGER - 1);
			System.out.println("Checking the addition of 1 and 2147483646 passed.");
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
			
		System.out.println();
		
	}
	
	
	/**
	 * testCheckTheAdditionOfForNarrowestInvalidRangeOfPositiveIntegers tests
	 * checkTheAdditionOfForNarrowestInvalidRangeOfPositiveIntegers by throwing an integer-overflow exception when
	 * the addition of 1 and 2147483647 is attempted.
	 */
	@Test
	public void testCheckTheAdditionOfForNarrowestInvalidRangeOfPositiveIntegers() {
		
		System.out.println("Running testCheckTheAdditionOfForNarrowestInvalidRangeOfPositiveIntegers.");
		
		try {
			checkTheAdditionOf(1, this.THE_MAXIMUM_INTEGER);
			System.out.println("Checking the addition of 1 and 2147483647 passed.");
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
			
		System.out.println();
		
	}
	

	/**
	 * testCheckTheAdditionOfForWidestValidRangeOfNegativeIntegers tests
	 * checkTheAdditionOfForWidestValidRangeOfNegativeIntegers by passing a check of the addition of -1 and -2147483647
	 */
	@Test
	public void testCheckTheAdditionOfForWidestValidRangeOfNegativeIntegers() {
		
		System.out.println("Running testCheckTheAdditionOfForWidestValidRangeOfNegativeIntegers.");
		
		try {
			checkTheAdditionOf(-1, this.THE_MINIMUM_INTEGER + 1);
			System.out.println("Checking the addition of -1 and -2147483647 passed.");
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
			
		System.out.println();
		
	}
	
	
	/**
	 * testCheckTheAdditionOfForNarrowestInvalidRangeOfPositiveIntegers tests
	 * checkTheAdditionOfForNarrowestInvalidRangeOfPositiveIntegers by throwing an integer-overflow exception when
	 * the addition of -1 and -2147483648 is attempted.
	 */
	@Test
	public void testCheckTheAdditionOfForNarrowestInvalidRangeOfNegativeIntegers() {
		
		System.out.println("Running testCheckTheAdditionOfForNarrowestInvalidRangeOfNegativeIntegers.");
		
		try {
			checkTheAdditionOf(-1, this.THE_MINIMUM_INTEGER);
			System.out.println("Checking the addition of -1 and -2147483648 passed.");
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
			
		System.out.println();
		
	}
	
	
}