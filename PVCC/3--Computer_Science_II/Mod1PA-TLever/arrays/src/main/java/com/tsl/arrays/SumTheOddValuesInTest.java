package com.tsl.more_operations_with_arrays;


import org.junit.jupiter.api.Test;


/**
 * SumTheOddValuesInTest encapsulates JUnit tests of core functionality of the method sumTheOddValuesIn of class App.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
class SumTheOddValuesInTest {

	
	/**
	 * testSumTheOddValuesInWithAnArrayWithAnExtremelyWideRange tests sumTheOddValuesIn by displaying the
	 * main output for an array with an extremely wide range.
	 */
	@Test
	public void testSumTheOddValuesInWithAnArrayWithAnExtremelyWideRange() {
		
		System.out.println("Running testSumTheOddValuesInWithAnArrayWithAnExtremelyWideRange.");
        
        int[] theArrayOfRandomIntegers = new int[] {
        	-2147483648,
        	-2147483647,
        	-2147483646,
        	-2147483645,
        	 2147483647,
        	 2147483646,
        	 2147483645,
        	 2147483644,
        	 2147483643
        };
        
        try {
        	int theSumOfTheOddValuesInTheArray = App.sumTheOddValuesIn(theArrayOfRandomIntegers);
        	System.out.println("The sum of odd values in the array is: " + theSumOfTheOddValuesInTheArray);
        }
        catch (AnIntegerOverflowException theIntegerOverflowException) {
        	System.out.println(theIntegerOverflowException.getMessage());
        }
        catch (ASumDoesNotExistException theSumDoesNotExistException) {
        	System.out.println(theSumDoesNotExistException.getMessage());
        }
		
		System.out.println();
		
	}
	
	
	/**
	 * testSumTheOddValuesInWithAnArrayOfLargePositiveIntegers tests sumTheOddValuesIn by displaying the message of
	 * an integer overflow exception that occurs when very large positive integers are summed.
	 */
	@Test
	public void testSumTheOddValuesInWithAnArrayOfLargePositiveIntegers() {
	
		System.out.println("Running testSumTheOddValuesInWithAnArrayOfLargePositiveIntegers.");
		
		int[] theArrayOfRandomIntegers = new int[] {App.THE_MAXIMUM_INTEGER, App.THE_MAXIMUM_INTEGER};
		
        try {
        	int theSumOfTheOddValuesInTheArray = App.sumTheOddValuesIn(theArrayOfRandomIntegers);
        	System.out.println("The sum of odd values in the array is: " + theSumOfTheOddValuesInTheArray);
        }
        catch (AnIntegerOverflowException theIntegerOverflowException) {
        	System.out.println(theIntegerOverflowException.getMessage());
        }
        catch (ASumDoesNotExistException theSumDoesNotExistException) {
        	System.out.println(theSumDoesNotExistException.getMessage());
        }
		
        System.out.println();
        
	}
	
	
	/**
	 * testSumTheOddValuesInWithAnArrayOfLargeMagnitudeNegativeIntegers tests sumTheOddValuesIn by displaying the message of
	 * an integer overflow exception that occurs when very large-magnitude negative integers are summed.
	 */
	@Test
	public void testSumTheOddValuesInWithAnArrayOfLargeMagnitudeNegativeIntegers() {
	
		System.out.println("Running testSumTheOddValuesInWithAnArrayOfLargeMagnitudeNegativeIntegers.");
		
		int[] theArrayOfRandomIntegers = new int[] {App.THE_MINIMUM_INTEGER+1, App.THE_MINIMUM_INTEGER+1};
		
        try {
        	int theSumOfTheOddValuesInTheArray = App.sumTheOddValuesIn(theArrayOfRandomIntegers);
        	System.out.println("The sum of odd values in the array is: " + theSumOfTheOddValuesInTheArray);
        }
        catch (AnIntegerOverflowException theIntegerOverflowException) {
        	System.out.println(theIntegerOverflowException.getMessage());
        }
        catch (ASumDoesNotExistException theSumDoesNotExistException) {
        	System.out.println(theSumDoesNotExistException.getMessage());
        }
        
        System.out.println();
		
	}
	
	
	/**
	 * testSumTheOddValuesInWithAnArrayOfOneElement tests sumTheOddValuesIn by displaying the main output for an array
	 * with one element.
	 */
	@Test
	public void testSumTheOddValuesInWithAnArrayOfOneElement() {
		
		System.out.println("Running testSumTheOddValuesInWithAnArrayOfOneElement.");
		
		int[] theArrayOfRandomIntegers = new int[] {1};
		
        try {
        	int theSumOfTheOddValuesInTheArray = App.sumTheOddValuesIn(theArrayOfRandomIntegers);
        	System.out.println("The sum of odd values in the array is: " + theSumOfTheOddValuesInTheArray);
        }
        catch (AnIntegerOverflowException theIntegerOverflowException) {
        	System.out.println(theIntegerOverflowException.getMessage());
        }
        catch (ASumDoesNotExistException theSumDoesNotExistException) {
        	System.out.println(theSumDoesNotExistException.getMessage());
        }
		
		System.out.println();
		
	}
	
	
	/**
	 * testSumTheOddValuesInWithAnArrayOfZeroElements tests sumTheOddValuesIn by displaying the message of a sum does
	 * not exist exception thrown when an array meant to have random integers is empty.
	 */
	@Test
	public void testSumTheOddValuesInWithAnArrayOfZeroElements() {
		
		System.out.println("Running testSumTheOddValuesInWithAnArrayOfZeroElements.");
		
		int[] theArrayOfRandomIntegers = new int[] {};
		
        try {
        	int theSumOfTheOddValuesInTheArray = App.sumTheOddValuesIn(theArrayOfRandomIntegers);
        	System.out.println("The sum of odd values in the array is: " + theSumOfTheOddValuesInTheArray);
        }
        catch (AnIntegerOverflowException theIntegerOverflowException) {
        	System.out.println(theIntegerOverflowException.getMessage());
        }
        catch (ASumDoesNotExistException theSumDoesNotExistException) {
        	System.out.println(theSumDoesNotExistException.getMessage());
        }
		
		System.out.println();
		
	}
	
}