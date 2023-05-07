package com.tsl.more_operations_with_arrays;


import org.junit.jupiter.api.Test;


/**
 * CalculateTheArrayOfTheSquaresOfTheValuesInTest encapsulates JUnit tests of core functionality of the method
 * calculateTheArrayOfTheSquaresOfTheValuesIn of class App.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
public class CalculateTheArrayOfTheSquaresOfTheValuesInTest {
	
	
	/**
	 * testCalculateTheArrayOfTheSquaresOfTheValuesInForTheWidestValidCenteredRange tests
	 * calculateTheArrayOfTheSquaresOfTheValuesIn by displaying the main output for an array of random
	 * integers with a range of -46,340 to 46,340 inclusive.
	 */
	@Test
	public void testCalculateTheArrayOfTheSquaresOfTheValuesInForTheWidestValidCenteredRange() {
		
		System.out.println("Running testCalculateTheArrayOfTheSquaresOfTheValuesInForTheWidestValidCenteredRange.");
		
        try {
    		int[] theArrayOfRandomIntegers = setUpAnArrayOfRandomIntegersGiven(-46340, 46340);
        	
        	int[] theArrayOfSquares = App.calculateTheArrayOfTheSquaresOfTheValuesIn(theArrayOfRandomIntegers);
        	System.out.println("The array of squares has the following values:");
        	App.display(theArrayOfSquares);
        }
        catch (AnIntegerOverflowException theIntegerOverflowException) {
        	System.out.println(theIntegerOverflowException.getMessage());
        }
		
		System.out.println();
		
	}
	
	
	/**
	 * testCalculateTheArrayOfTheSquaresOfTheValuesInForAnArrayOfValuesTooSmall tests
	 * calculateTheArrayOfTheSquaresOfTheValuesIn by displaying the message of an integer overflow exception thrown
	 * when a value in an array of random integers is too small to be squared.
	 */
	@Test
	public void testCalculateTheArrayOfTheSquaresOfTheValuesInForAnArrayOfValuesTooSmall() {
		
		System.out.println("Running testCalculateTheArrayOfTheSquaresOfTheValuesInForAnArrayOfValuesTooSmall.");
		
        try {
    		int[] theArrayOfRandomIntegers = setUpAnArrayOfRandomIntegersGiven(-46341, -46341);
        	
        	int[] theArrayOfSquares = App.calculateTheArrayOfTheSquaresOfTheValuesIn(theArrayOfRandomIntegers);
        	System.out.println("The array of squares has the following values:");
        	App.display(theArrayOfSquares);
        }
        catch (AnIntegerOverflowException theIntegerOverflowException) {
        	System.out.println(theIntegerOverflowException.getMessage());
        }
		
		System.out.println();
		
	}
	
	
	/**
	 * testCalculateTheArrayOfTheSquaresOfTheValuesInForAnArrayOfValuesTooLarge tests
	 * calculateTheArrayOfTheSquaresOfTheValuesIn by displaying the message of an integer overflow exception thrown
	 * when a value in an array of random integers is too large to be squared.
	 */
	@Test
	public void testCalculateTheArrayOfTheSquaresOfTheValuesInForAnArrayOfValuesTooLarge() {
		
		System.out.println("Running testCalculateTheArrayOfTheSquaresOfTheValuesInForAnArrayOfValuesTooLarge.");
		
        try {
    		int[] theArrayOfRandomIntegers = setUpAnArrayOfRandomIntegersGiven(46341, 46341);
        	
        	int[] theArrayOfSquares = App.calculateTheArrayOfTheSquaresOfTheValuesIn(theArrayOfRandomIntegers);
        	System.out.println("The array of squares has the following values:");
        	App.display(theArrayOfSquares);
        }
        catch (AnIntegerOverflowException theIntegerOverflowException) {
        	System.out.println(theIntegerOverflowException.getMessage());
        }
		
		System.out.println();
		
	}
	
	
	private int[] setUpAnArrayOfRandomIntegersGiven(int theLowerLimit, int theUpperLimit)
		throws AnIntegerOverflowException {
		
		int[] theArrayOfRandomIntegers = new int[1000];
		
        int[] theLowerLimitAndTheUpperLimitForAnInteger = new int[] {theLowerLimit, theUpperLimit};
        
        ARandomNumberGenerator theRandomNumberGenerator = new ARandomNumberGenerator();
        
        for (int i = 0; i < theArrayOfRandomIntegers.length; i++) {
        	theArrayOfRandomIntegers[i] =
        		theRandomNumberGenerator.getARandomIntegerInclusivelyBetween(
        			theLowerLimitAndTheUpperLimitForAnInteger[0], theLowerLimitAndTheUpperLimitForAnInteger[1]);
        }
        
        return theArrayOfRandomIntegers;
		
	}
	
}
