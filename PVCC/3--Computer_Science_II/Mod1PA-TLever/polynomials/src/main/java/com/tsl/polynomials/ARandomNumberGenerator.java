package com.tsl.polynomials;


import java.util.Random;


/**
 * ARandomNumberGenerator represents the structure for some random number generators.
 * @author Tom Lever
 * @version 1.0
 * @since 05/17/21
 *
 */
class ARandomNumberGenerator {
	
	
	/**
	 * thePopularRandomNumberGenerator is a component of ARandomNumberGenerator.
	 */
	private Random thePopularRandomNumberGenerator;
	
	
	/**
	 * ARandomNumberGenerator() is the zero-argument constructor for ARandomNumberGenerator.
	 */
	public ARandomNumberGenerator() {
		
		thePopularRandomNumberGenerator = new Random();
		
	}
	
	
	/**
	 * getARandomIntegerInclusivelyBetween provides a integer between a lower limit and an upper limit inclusive.
	 * @param theLowerLimit
	 * @param theUpperLimit
	 * @return
	 */
    public int getARandomIntegerInclusivelyBetween(int theLowerLimit, int theUpperLimit)
    	throws AnIntegerOverflowException {
    	
		if (theUpperLimit > PolynomialDriver.THE_MAXIMUM_INTEGER + theLowerLimit - 1) {
			throw new AnIntegerOverflowException(
				"The proposed range [lower limit, upper limit] is too wide for a random number generator.\n" +
				"A possible range is [-1,073,741,823, 1,073,741,823]."
			);
		}
    	
    	return this.thePopularRandomNumberGenerator.nextInt((theUpperLimit - theLowerLimit) + 1) + theLowerLimit;
    }
	
}