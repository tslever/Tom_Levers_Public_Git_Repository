package com.tsl.pez_candy;


import org.junit.jupiter.api.Test;


/**
 * ReceivesTest encapsulates JUnit tests of core functionality of the method receives of class PezCandy.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */

public class ReceivesTest {

	
	/**
	 * testReceives tests receives by setting up an empty Pez-candy dispenser and pushing the maximum possible number of
	 * candies to it.
	 */
	@Test
	public void testReceivesWhileDispenserIsNotFull() {
		
		System.out.println("Running testReceivesWhileDispenserIsNotFull.");
		
		PezCandy theSparePezCandyDispenser = new PezCandy("The spare Pez-candy dispenser");
		
		try {
			
	        while (!theSparePezCandyDispenser.isFull()) {
	            theSparePezCandyDispenser.receives(new Candy(AColor.getARandomColor()));
	        }
	        
	        System.out.println("The spare Pez-candy dispenser received candies with random colors until it was full.");
			
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
		
		catch (AStackOverflowException theStackOverflowException) {
			System.out.println(theStackOverflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	/**
	 * testReceives tests receives by setting up an empty Pez-candy dispenser, pushing the maximum possible number of
	 * candies to it, and trying to push one more.
	 */
	@Test
	public void testReceivesTooLong() {
		
		System.out.println("Running testReceivesTooLong.");
		
		PezCandy theSparePezCandyDispenser = new PezCandy("The spare Pez-candy dispenser");
		
		try {
			
	        for (int i = 0; i <= theSparePezCandyDispenser.THE_NUMBER_OF_PEZ_CANDIES_IN_A_NEW_DISPENSER; i++) {
	            theSparePezCandyDispenser.receives(new Candy(AColor.getARandomColor()));
	        }
	        
	        System.out.println("The spare Pez-candy dispenser received candies with random colors until it was full.");
			
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
		
		catch (AStackOverflowException theStackOverflowException) {
			System.out.println(theStackOverflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
}
