package com.tsl.polynomials;


import org.junit.jupiter.api.Test;


/**
 * CheckTheMultiplicationOfTest encapsulates JUnit tests of core functionality of the method checkTheMultiplicationOf
 * of class Polynomial.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
public class CheckTheMultiplicationOfTest {

	
	/**
	 * testCheckTheMultiplicationOfForValidSamedSignedIntegers tests checkTheMultiplicationOf for two valid same-signed
	 * integers whose product is less than the maximum integer.
	 */
	@Test
	public void testCheckTheMultiplicationOfForValidSameSignedIntegers() {
		
		System.out.println("Running testCheckTheMultiplicationOfForValidSameSignedIntegers");
		
		int theFirstInteger = -46340;
		int theSecondInteger = -46341;
		
		
		try {
		
			if (((theFirstInteger > 0) && (theSecondInteger > 0) && (theFirstInteger > PolynomialDriver.THE_MAXIMUM_INTEGER / theSecondInteger)) ||
				((theFirstInteger < 0) && (theSecondInteger > 0) && (theFirstInteger < PolynomialDriver.THE_MINIMUM_INTEGER / theSecondInteger)) ||
				((theFirstInteger > 0) && (theSecondInteger < 0) && (theFirstInteger > PolynomialDriver.THE_MINIMUM_INTEGER / theSecondInteger)) ||
				((theFirstInteger < 0) && (theSecondInteger < 0) && (theFirstInteger < PolynomialDriver.THE_MAXIMUM_INTEGER / theSecondInteger))) {
				
				throw new AnIntegerOverflowException(
					"Integer-overflow exception: the product of " +
					theFirstInteger +
					" and " +
					theSecondInteger +
					" exceeds " +
					PolynomialDriver.THE_MAXIMUM_INTEGER +
					"."
				);
				
			}
			
			System.out.println("The product: " + theFirstInteger * theSecondInteger);
		
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
	
		System.out.println();
		
	}
	
	
	/**
	 * testCheckTheMultiplicationOfForInvalidSameSignedIntegers tests checkTheMultiplicationOf for two same-signed
	 * integers whose product is greater than the maximum integer.
	 */
	@Test
	public void testCheckTheMultiplicationOfForInvalidSameSignedIntegers() {
		
		System.out.println("Running testCheckTheMultiplicationOfForInvalidSameSignedIntegers");
		
		int theFirstInteger = -46340;
		int theSecondInteger = -46342;
		
		
		try {
		
			if (((theFirstInteger > 0) && (theSecondInteger > 0) && (theFirstInteger > PolynomialDriver.THE_MAXIMUM_INTEGER / theSecondInteger)) ||
				((theFirstInteger < 0) && (theSecondInteger > 0) && (theFirstInteger < PolynomialDriver.THE_MINIMUM_INTEGER / theSecondInteger)) ||
				((theFirstInteger > 0) && (theSecondInteger < 0) && (theFirstInteger > PolynomialDriver.THE_MINIMUM_INTEGER / theSecondInteger)) ||
				((theFirstInteger < 0) && (theSecondInteger < 0) && (theFirstInteger < PolynomialDriver.THE_MAXIMUM_INTEGER / theSecondInteger))) {
				
				throw new AnIntegerOverflowException(
					"Integer-overflow exception: the product of " +
					theFirstInteger +
					" and " +
					theSecondInteger +
					" exceeds " +
					PolynomialDriver.THE_MAXIMUM_INTEGER +
					"."
				);
				
			}
			
			System.out.println("The product: " + theFirstInteger * theSecondInteger);
		
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
	
		System.out.println();
		
	}
	
	
	/**
	 * testCheckTheMultiplicationOfForValidMixedIntegers tests checkTheMultiplicationOf for two mixed-sign integers
	 * whose product is greater than the minimum integer.
	 */
	@Test
	public void testCheckTheMultiplicationOfForValidMixedIntegers() {
		
		System.out.println("Running testCheckTheMultiplicationOfForValidMixedIntegers");
		
		int theFirstInteger = -46341;
		int theSecondInteger = 46340;
		
		
		try {
		
			if (((theFirstInteger > 0) && (theSecondInteger > 0) && (theFirstInteger > PolynomialDriver.THE_MAXIMUM_INTEGER / theSecondInteger)) ||
				((theFirstInteger < 0) && (theSecondInteger > 0) && (theFirstInteger < PolynomialDriver.THE_MINIMUM_INTEGER / theSecondInteger)) ||
				((theFirstInteger > 0) && (theSecondInteger < 0) && (theFirstInteger > PolynomialDriver.THE_MINIMUM_INTEGER / theSecondInteger)) ||
				((theFirstInteger < 0) && (theSecondInteger < 0) && (theFirstInteger < PolynomialDriver.THE_MAXIMUM_INTEGER / theSecondInteger))) {
				
				throw new AnIntegerOverflowException(
					"Integer-overflow exception: the product of " +
					theFirstInteger +
					" and " +
					theSecondInteger +
					" exceeds " +
					PolynomialDriver.THE_MAXIMUM_INTEGER +
					"."
				);
				
			}
			
			System.out.println("The product: " + theFirstInteger * theSecondInteger);
		
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
	
		System.out.println();
		
	}
	
	
	/**
	 * testCheckTheMultiplicationOfForInvalidMixedIntegers tests checkTheMultiplicationOf for two mixed-sign integers
	 * whose product is less than the minimum integer.
	 */
	@Test
	public void testCheckTheMultiplicationOfForInvalidMixedIntegers() {
		
		System.out.println("Running testCheckTheMultiplicationOfForInvalidMixedIntegers");
		
		int theFirstInteger = -46340;
		int theSecondInteger = 46342;
		
		
		try {
		
			if (((theFirstInteger > 0) && (theSecondInteger > 0) && (theFirstInteger > PolynomialDriver.THE_MAXIMUM_INTEGER / theSecondInteger)) ||
				((theFirstInteger < 0) && (theSecondInteger > 0) && (theFirstInteger < PolynomialDriver.THE_MINIMUM_INTEGER / theSecondInteger)) ||
				((theFirstInteger > 0) && (theSecondInteger < 0) && (theFirstInteger > PolynomialDriver.THE_MINIMUM_INTEGER / theSecondInteger)) ||
				((theFirstInteger < 0) && (theSecondInteger < 0) && (theFirstInteger < PolynomialDriver.THE_MAXIMUM_INTEGER / theSecondInteger))) {
				
				throw new AnIntegerOverflowException(
					"Integer-overflow exception: the product of " +
					theFirstInteger +
					" and " +
					theSecondInteger +
					" exceeds " +
					PolynomialDriver.THE_MAXIMUM_INTEGER +
					"."
				);
				
			}
			
			System.out.println("The product: " + theFirstInteger * theSecondInteger);
		
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
	
		System.out.println();
		
	}
	
	
}
