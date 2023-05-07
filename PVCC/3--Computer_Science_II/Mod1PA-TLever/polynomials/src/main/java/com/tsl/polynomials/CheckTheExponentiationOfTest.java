package com.tsl.polynomials;


import org.junit.jupiter.api.Test;


/**
 * CheckTheExponentiationOfTest encapsulates JUnit tests of core functionality of the method checkTheExponentiationOf
 * of class Polynomial.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
public class CheckTheExponentiationOfTest {

	
	/**
	 * testCheckTheExponentiationOfForNonNegativePower tests checkTheMultiplicationOf for a non-negative power.
	 * code of exponentiate is extracted here so that creating a Polynomial object may be avoided.
	 */
	@Test
	public void testCheckTheExponentiationOfForNonNegativePower() {
		
		System.out.println("Running testCheckTheExponentiationOfForNonNegativePower.");
		
		int theBase = 3;
		
		int thePower = 0;
		
		int theProduct;
		
		try {
			
			if (thePower < 0) {
				throw new ANotSufficientlyImplementedException(
					"Exponentiation with negative integer powers has not been implemented yet.");
			}
			
			if (thePower == 0) {
				theProduct = 1;
				System.out.println(
					"The product of exponentiating " + theBase + " to " + thePower + " is  " + theProduct + "."
				);
			}
			
		}
		
		catch (ANotSufficientlyImplementedException theNotSufficientlyImplementedException) {
			System.out.println(theNotSufficientlyImplementedException.getMessage());
		}
	
		System.out.println();
		
	}
	
	
	/**
	 * testCheckTheExponentiationOfForNegativePower tests checkTheMultiplicationOf for a negative power, and outputs the
	 * message of ANotSufficientlyImplementedException occurs because functionality to handle negative powers has not
	 * been implemented yet.
	 */
	@Test
	public void testCheckTheExponentiationOfForNegativePower() {
		
		System.out.println("Running testCheckTheExponentiationOfForNonNegativePower.");
		
		int theBase = 3;
		
		int thePower = -1;
		
		int theProduct;
		
		try {
			
			if (thePower < 0) {
				throw new ANotSufficientlyImplementedException(
					"Exponentiation with negative integer powers has not been implemented yet.");
			}
			
			if (thePower == 0) {
				theProduct = 1;
				System.out.println(
					"The product of exponentiating " + theBase + " to " + thePower + " is  " + theProduct + "."
				);
			}
			
		}
		
		catch (ANotSufficientlyImplementedException theNotSufficientlyImplementedException) {
			System.out.println(theNotSufficientlyImplementedException.getMessage());
		}
	
		System.out.println();
		
	}
	
}