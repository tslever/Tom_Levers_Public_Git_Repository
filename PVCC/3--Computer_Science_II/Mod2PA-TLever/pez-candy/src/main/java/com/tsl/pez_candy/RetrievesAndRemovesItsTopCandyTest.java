package com.tsl.pez_candy;


import org.junit.jupiter.api.Test;


/**
 * RetrievesAndRemovesItsTopCandyTest encapsulates JUnit tests of core functionality of the method
 * retrievesAndRemovesItsTopCandy of class PezCandy.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */

public class RetrievesAndRemovesItsTopCandyTest {

	
	/**
	 * testRetrieveAndRemovesItsTopCandyWhileDispenserIsNotEmpty tests retrievesAndRemovesItsTopCandy by setting up a
	 * full Pez-candy dispenser and retrieving and removing all candies from the dispenser.
	 */
	@Test
	public void testRetrievesAndRemovesItsTopCandyWhileDispenserIsNotEmpty() {
		
		System.out.println("Running testRetrievesAndRemovesItsTopCandyWhileDispenserIsNotEmpty.");
		
		try {
			AFullPezCandyDispenser todaysPezCandyDispenser = new AFullPezCandyDispenser("Today's Pez-candy dispenser");
			
	        while (!todaysPezCandyDispenser.isEmpty()) {
	            todaysPezCandyDispenser.retrievesAndRemovesItsTopCandy();
	        }
	        
	        System.out.println("Retrieved and removed all candies from todaysPezCandyDispenser.");
			
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
		
		catch (AStackUnderflowException theStackUnderflowException) {
			System.out.println(theStackUnderflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	/**
	 * testRetrieveAndRemovesItsTopCandyAsLongAsPossible tests retrievesAndRemovesItsTopCandy by setting up a full
	 * Pez-candy dispenser, retrieving and removing all candies from the dispenser, and trying to retreive and remove
	 * one more candy from the dispenser.
	 */
	@Test
	public void testRetrievesAndRemovesItsTopCandyTooLong() {
		
		System.out.println("Running testRetrievesAndRemovesItsTopCandyTooLong.");
		
		try {
			AFullPezCandyDispenser todaysPezCandyDispenser = new AFullPezCandyDispenser("Today's Pez-candy dispenser");
			
	        for (int i = 0; i <= todaysPezCandyDispenser.THE_NUMBER_OF_PEZ_CANDIES_IN_A_NEW_DISPENSER; i++) {
	            todaysPezCandyDispenser.retrievesAndRemovesItsTopCandy();
	        }
	        
	        System.out.println("Retrieved and removed all candies from todaysPezCandyDispenser.");
			
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
		
		catch (AStackUnderflowException theStackUnderflowException) {
			System.out.println(theStackUnderflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
}
