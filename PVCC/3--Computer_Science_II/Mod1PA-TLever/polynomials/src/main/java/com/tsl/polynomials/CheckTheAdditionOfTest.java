package com.tsl.polynomials;


import org.junit.jupiter.api.Test;


/**
 * CheckTheAdditionnOfTest encapsulates JUnit tests of core functionality of the method checkTheAdditionOf of class
 * Polynomial.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
public class CheckTheAdditionOfTest {

	
	/**
	 * testCheckTheAdditionOfTwoValidIntegers tests checkTheAdditionOf for the closest differently signed integers. 
	 */
	@Test
	public void testCheckTheAdditionOfForTheClosestDifferentlySignedIntegers() {
		
		System.out.println("Running testCheckTheAdditionOfForTheClosestDifferentlySignedIntegers");
		
		int theFirstInteger = PolynomialDriver.THE_MINIMUM_INTEGER;
		int theSecondInteger = 0;
		
		try {
		
			if ((theFirstInteger > 0) && (theSecondInteger > 0) && (theFirstInteger > PolynomialDriver.THE_MAXIMUM_INTEGER - theSecondInteger) ||
				(theFirstInteger < 0) && (theSecondInteger < 0) && (theFirstInteger < PolynomialDriver.THE_MINIMUM_INTEGER - theSecondInteger)) {
				
				throw new AnIntegerOverflowException(
					"Integer-overflow exception: the sum of " +
					theFirstInteger +
					" and " +
					theSecondInteger +
					" is outside the interval [" +
					PolynomialDriver.THE_MINIMUM_INTEGER +
					", " +
					PolynomialDriver.THE_MAXIMUM_INTEGER +
					"]."
				);
				
			}
	
			System.out.println(
				"The sum of integers " +
				theFirstInteger +
				" and " +
				theSecondInteger +
				" is " +
				theFirstInteger + theSecondInteger +
				"."
			);
		
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	/**
	 * testCheckTheAdditionOfTwoValidIntegers tests checkTheAdditionOf for the farthest same-signed integers. 
	 */
	@Test
	public void testCheckTheAdditionOfForTheFarthestSameSignedIntegers() {
		
		System.out.println("Running testCheckTheAdditionOfForTheFarthestSameSignedIntegers.");
		
		int theFirstInteger = PolynomialDriver.THE_MINIMUM_INTEGER;
		int theSecondInteger = -1;
		
		try {
		
			if ((theFirstInteger > 0) && (theSecondInteger > 0) && (theFirstInteger > PolynomialDriver.THE_MAXIMUM_INTEGER - theSecondInteger) ||
				(theFirstInteger < 0) && (theSecondInteger < 0) && (theFirstInteger < PolynomialDriver.THE_MINIMUM_INTEGER - theSecondInteger)) {
				
				throw new AnIntegerOverflowException(
					"Integer-overflow exception: the sum of " +
					theFirstInteger +
					" and " +
					theSecondInteger +
					" is outside the interval [" +
					PolynomialDriver.THE_MINIMUM_INTEGER +
					", " +
					PolynomialDriver.THE_MAXIMUM_INTEGER +
					"]."
				);
				
			}
	
			System.out.println(
				"The sum of integers " +
				theFirstInteger +
				" and " +
				theSecondInteger +
				" is " +
				theFirstInteger + theSecondInteger +
				"."
			);
		
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
}