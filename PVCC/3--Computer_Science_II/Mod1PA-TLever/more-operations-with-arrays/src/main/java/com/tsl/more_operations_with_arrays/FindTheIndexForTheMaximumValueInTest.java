package com.tsl.more_operations_with_arrays;


import org.junit.jupiter.api.Test;


/**
 * FindTheIndexForTheMaximumValueInTest encapsulates JUnit tests of core functionality of the method
 * findTheIndexForTheMaximumValueIn of class App.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
public class FindTheIndexForTheMaximumValueInTest {

	
	/**
	 * testFindTheIndexForTheMaximumValueInForANonEmptyArray tests findTheIndexForTheMaximumValueIn by displaying the
	 * main output for a non-empty array.
	 */
	@Test
	public void testFindTheIndexForTheMaximumValueInForANonEmptyArray() {
		
		System.out.println("Running testFindTheIndexForTheMaximumValueInForANonEmptyArray.");
		
		int[] theArrayOfRandomIntegers = new int[] {1};
		
		try {
			int theIndexOfTheMaximumValue = App.findTheIndexForTheMaximumValueIn(theArrayOfRandomIntegers);
			System.out.println("The index of the maximum value is: " + theIndexOfTheMaximumValue);
		}
		catch (AMaximumValueDoesNotExistException theMaximumValueDoesNotExistException) {
			System.out.println(theMaximumValueDoesNotExistException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	/**
	 * testFindTheIndexForTheMaximumValueInForANonEmptyArray tests findTheIndexForTheMaximumValueIn by displaying the
	 * message of a maximum value does not exist exception thrown when an array meant to be of random integers is empty.
	 */
	@Test
	public void testFindTheIndexForTheMaximumValueInForAnEmptyArray() {
		
		System.out.println("Running testFindTheIndexForTheMaximumValueInForAnEmptyArray.");
		
		int[] theArrayOfRandomIntegers = new int[] {};
		
		try {
			int theIndexOfTheMaximumValue = App.findTheIndexForTheMaximumValueIn(theArrayOfRandomIntegers);
			System.out.println("The index of the maximum value is: " + theIndexOfTheMaximumValue);
		}
		catch (AMaximumValueDoesNotExistException theMaximumValueDoesNotExistException) {
			System.out.println(theMaximumValueDoesNotExistException.getMessage());
		}
		
		System.out.println();
		
	}
	
}
