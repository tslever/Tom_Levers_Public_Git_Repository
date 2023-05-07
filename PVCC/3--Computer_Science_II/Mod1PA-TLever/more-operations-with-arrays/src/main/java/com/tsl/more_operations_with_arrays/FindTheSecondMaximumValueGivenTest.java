package com.tsl.more_operations_with_arrays;


import org.junit.jupiter.api.Test;


/**
 * FindTheSecondMaximumValueGivenTest encapsulates JUnit tests of core functionality of the method
 * findTheSecondMaximumValueGiven of class App.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
public class FindTheSecondMaximumValueGivenTest {

	
	/**
	 * testFindTheSecondMaximumValueGivenForAnArrayWithTwoElements tests findTheSecondMaximumValueGiven by displaying the
	 * main output for an array with two elements.
	 */
	@Test
	public void testFindTheSecondMaximumValueGivenForAnArrayWithTwoElements() {
		
		System.out.println("Running testFindTheSecondMaximumValueGivenForAnArrayWithTwoElements.");
		
		int[] theArrayOfRandomIntegers = {1, 2};
		
		try {
			int theSecondMaximumValue = App.findTheSecondMaximumValueGiven(theArrayOfRandomIntegers, 1);
			System.out.println("The second maximum value is: " + theSecondMaximumValue);
		}
		catch (ASecondMaximumValueDoesNotExistException theSecondMaximumValueDoesNotExistException) {
			System.out.println(theSecondMaximumValueDoesNotExistException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	/**
	 * testFindTheSecondMaximumValueGivenForAnArrayWithOneElement tests findTheSecondMaximumValueGiven by displaying the
	 * message of a second maximum value does not exist exception thrown when an array of random integers has only one
	 * element.
	 */
	@Test
	public void testFindTheSecondMaximumValueGivenForAnArrayWithOneElement() {
		
		System.out.println("Running testFindTheSecondMaximumValueGivenForAnArrayWithOneElement.");
		
		int[] theArrayOfRandomIntegers = {1};
		
		try {
			int theSecondMaximumValue = App.findTheSecondMaximumValueGiven(theArrayOfRandomIntegers, 0);
			System.out.println("The second maximum value is: " + theSecondMaximumValue);
		}
		catch (ASecondMaximumValueDoesNotExistException theSecondMaximumValueDoesNotExistException) {
			System.out.println(theSecondMaximumValueDoesNotExistException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
}
