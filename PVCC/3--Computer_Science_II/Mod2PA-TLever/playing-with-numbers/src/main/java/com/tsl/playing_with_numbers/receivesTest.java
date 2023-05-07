package com.tsl.playing_with_numbers;


import org.junit.jupiter.api.Test;


/**
 * receivesTest encapsulates JUnit tests of core functionality of the method receives of class
 * AnArrayListBasedBoundedStack.
 * @author Tom Lever
 * @version 1.0
 * @since 05/29/21
 *
 */
public class receivesTest {

	
	/**
	 * testReceivesByFillingToCapacity outputs a status update after an ArrayList-based bounded stack is filled to
	 * capacity.
	 */
	@Test
	public void testReceivesByFillingToCapacity() {
	
		System.out.println("Running testReceivesByFillingToCapacity.");
		
		AnArrayListBasedBoundedStack<Integer> theArrayListBasedBoundedStack =
			new AnArrayListBasedBoundedStack<Integer>(3);
		
		try {
		
			for (int i = 0; i < 3; i++) {
				theArrayListBasedBoundedStack.receives(i);
			}
			
			System.out.println("An ArrayList-based bounded stack was filled to capacity.");
			
		}
		
		catch (AStackOverflowException theStackOverflowException) {
			System.out.println(theStackOverflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	/**
	 * testReceivesByTryingToOverfill outputs the message of a stack-overflow exception that occurs after an
	 * ArrayList-based bounded stack is filled to capacity, and the test tries to add one more integer to the stack.
	 */
	@Test
	public void testReceivesByTryingToOverfill() {
	
		System.out.println("Running testReceivesByTryingToOverfill.");
		
		AnArrayListBasedBoundedStack<Integer> theArrayListBasedBoundedStack =
			new AnArrayListBasedBoundedStack<Integer>(3);
		
		try {
		
			for (int i = 0; i <= 3; i++) {
				theArrayListBasedBoundedStack.receives(i);
			}

			System.out.println("An ArrayList-based bounded stack was filled to capacity.");
			
		}
		
		catch (AStackOverflowException theStackOverflowException) {
			System.out.println(theStackOverflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
}
