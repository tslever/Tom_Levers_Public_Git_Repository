package com.tsl.playing_with_numbers;


import org.junit.jupiter.api.Test;


/**
 * mainTest encapsulates JUnit tests of core functionality of the method main of class App.
 * @author Tom Lever
 * @version 1.0
 * @since 05/29/21
 *
 */
public class mainTest {

	
	/**
	 * testMainWithStackOfOneInteger outputs the minimum integer in a stack of one integer.
	 */
	@Test
	public void testMainWithStackOfOneInteger() {
		
		System.out.println("Running testMainWithStackOfOneInteger.");
		
		try {
		
	        AFullArrayListBasedBoundedStackOfRandomIntegers theFullArrayListBasedBoundedStackOfRandomIntegers =
	        	new AFullArrayListBasedBoundedStackOfRandomIntegers(1);
	        
	        System.out.println("The integers in the stack are:\n" + theFullArrayListBasedBoundedStackOfRandomIntegers);
	        
	        System.out.println(
	        	"The minimum integer in the stack is: " +
	        	theFullArrayListBasedBoundedStackOfRandomIntegers.getTheMinimumInteger()
	        );
        
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
		
		catch (AStackOverflowException theStackOverflowException) {
			System.out.println(theStackOverflowException.getMessage());
		}
		
		catch (ANoMinimumIntegerExistsException theNoMinimumIntegerExistsException) {
			System.out.println(theNoMinimumIntegerExistsException.getMessage());
		}
        
        System.out.println();
		
	}
	
	
	/**
	 * testMainWithStackOfZeroIntegers outputs the message of a no minimum integer exists exception that occurs when
	 * the minimum integer of an empty stack is requested.
	 */
	@Test
	public void testMainWithStackOfZeroIntegers() {
		
		System.out.println("Running testMainWithStackOfZeroIntegers.");
		
		try {
		
	        AFullArrayListBasedBoundedStackOfRandomIntegers theFullArrayListBasedBoundedStackOfRandomIntegers =
	        	new AFullArrayListBasedBoundedStackOfRandomIntegers(0);
	        
	        System.out.println("The integers in the stack are:\n" + theFullArrayListBasedBoundedStackOfRandomIntegers);
	        
	        System.out.println(
	        	"The minimum integer in the stack is: " +
	        	theFullArrayListBasedBoundedStackOfRandomIntegers.getTheMinimumInteger()
	        );
        
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
		
		catch (AStackOverflowException theStackOverflowException) {
			System.out.println(theStackOverflowException.getMessage());
		}
		
		catch (ANoMinimumIntegerExistsException theNoMinimumIntegerExistsException) {
			System.out.println(theNoMinimumIntegerExistsException.getMessage());
		}
        
        System.out.println();
		
	}
	
	
	/**
	 * testThrowingANoMaximumIntegerExistsException outputs the message of a no maximum integer exists exception that
	 * occurs when the maximum integer of an empty stack is requested.
	 */
	@Test
	public void testThrowingANoMaximumIntegerExistsException() {
		
		System.out.println("Running testThrowingANoMaximumIntegerExistsException.");
		
		try {
		
	        AFullArrayListBasedBoundedStackOfRandomIntegers theFullArrayListBasedBoundedStackOfRandomIntegers =
	        	new AFullArrayListBasedBoundedStackOfRandomIntegers(0);
	        
	        System.out.println("The integers in the stack are:\n" + theFullArrayListBasedBoundedStackOfRandomIntegers);
	        
	        System.out.println(
            	"The maximum integer in the stack is: " +
            	theFullArrayListBasedBoundedStackOfRandomIntegers.getTheMaximumInteger()
            );
        
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
		
		catch (AStackOverflowException theStackOverflowException) {
			System.out.println(theStackOverflowException.getMessage());
		}
		
		catch (ANoMaximumIntegerExistsException theNoMaximumIntegerExistsException) {
			System.out.println(theNoMaximumIntegerExistsException.getMessage());
		}
        
        System.out.println();
		
	}
	
	
	/**
	 * testThrowingANoAverageExistsException outputs the message of a no minimum integer exists exception that occurs
	 * when the average of integers in an empty stack is requested.
	 */
	@Test
	public void testThrowingANoAverageExistsException() {
		
		System.out.println("Running testThrowingANoAverageExistsException.");
		
		try {
		
	        AFullArrayListBasedBoundedStackOfRandomIntegers theFullArrayListBasedBoundedStackOfRandomIntegers =
	        	new AFullArrayListBasedBoundedStackOfRandomIntegers(0);
	        
	        System.out.println("The integers in the stack are:\n" + theFullArrayListBasedBoundedStackOfRandomIntegers);
	        
	        System.out.println(
            	"The average of the integers in the stack is: " +
            	theFullArrayListBasedBoundedStackOfRandomIntegers.getTheAverageOfTheIntegers()
            );
        
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
		
		catch (AStackOverflowException theStackOverflowException) {
			System.out.println(theStackOverflowException.getMessage());
		}
		
		catch (ANoAverageExistsException theNoAverageExistsException) {
			System.out.println(theNoAverageExistsException.getMessage());
		}
        
        System.out.println();
		
	}
	
}
