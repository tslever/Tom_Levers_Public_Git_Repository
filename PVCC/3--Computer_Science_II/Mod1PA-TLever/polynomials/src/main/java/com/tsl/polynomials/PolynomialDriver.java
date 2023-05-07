package com.tsl.polynomials;

import java.util.Arrays;


/**
 * PolynomialDriver encapsulates the entry point to this program that performs algebra with polynomials.
 * @author Tom Lever
 * @version 1.0
 * @since 05/22/21
 *
 */
class PolynomialDriver {
	
	
	/**
	 * THE_MINIMUM_INTEGER is an attribute of PolynomialDriver.
	 */
	public static final int THE_MINIMUM_INTEGER = -2147483648;
	
	
	/**
	 * THE_MAXIMUM_INTEGER is an attribute of PolynomialDriver.
	 */
	public static final int THE_MAXIMUM_INTEGER = 2147483647;
	
	
	/**
	 * main in the entry point to this program that performs algebra with polynomials.
	 * @param args
	 * @throws AnInvalidDegreeException
	 * @throws AnInvalidArrayOfCoefficientsGivenDegreeException
	 * @throws AnIntegerOverflowException
	 * @throws ANotSufficientlyImplementedException
	 */
    public static void main( String[] args )
    	throws
    		AnInvalidDegreeException,
    		AnInvalidArrayOfCoefficientsGivenDegreeException,
    		AnIntegerOverflowException,
    		ANotSufficientlyImplementedException {
        
    	Polynomial zero = new Polynomial(0, new int[] {0});
    	System.out.println(
    		"Created polynomial zero = "
    		+ zero +
    		" based on ultimate degree " +
    		zero.getDegree() +
    		" and array of coefficients " +
    		Arrays.toString(zero.getCoefficients())
    	);
    	
    	int theDegree = 4;
    	Polynomial p1 = new Polynomial(theDegree, createAnArrayOfRandomIntegersBasedOn(theDegree, 0, 10));
    	System.out.println(
    		"Created polynomial p1 = "
    		+ p1 +
    		" based on degree " +
    		p1.getDegree() +
    		" and array of random coefficients " +
    		Arrays.toString(p1.getCoefficients())
    	);
    	
    	
    	theDegree = 3;
    	Polynomial p2 = new Polynomial(theDegree, createAnArrayOfRandomIntegersBasedOn(theDegree, 0, 10));
    	System.out.println(
    		"Created polynomial p2 = "
    		+ p2 +
    		" based on degree " +
    		p2.getDegree() +
    		" and array of random coefficients " +
    		Arrays.toString(p2.getCoefficients())
    	);
    	
    	
    	System.out.println(
    		"p1 + p2 equals " +
    		p1.addPoly(p2) +
    		" and has coefficients " +
    		Arrays.toString(p1.addPoly(p2).getCoefficients())
    	);
    	
    	System.out.println(
    		"0 - p1 equals " +
    		zero.subtractPoly(p1) +
    		" and has coefficients " +
    		Arrays.toString(zero.subtractPoly(p1).getCoefficients())
    	);
    	
    	System.out.println("p1(x = 2) = " + p1.evaluate(2));
    	
    	if (p1.equals(p1)) {
    		System.out.println(p1 + " equals " + p1);
    	}
    	else {
    		System.out.println(p1 + " does not equal " + p1);
    	}
    	
    	if (p1.equals(p2)) {
    		System.out.println(p1 + " equals " + p2);
    	}
    	else {
    		System.out.println(p1 + " does not equal " + p2);
    	}
    	
    	System.out.println("The derivative of p1 = " + p1 + " is d(p1)/dx = " + p1.derivative());
    	
    }
    
    
    /**
     * createAnArrayOfRandomIntegers creates an array of random integers based on requests for parameters from a user.
     * @return
     */
    private static int[] createAnArrayOfRandomIntegersBasedOn(int theDegree, int theLowerLimit, int theUpperLimit)
    	throws AnIntegerOverflowException {
    	
        int[] theArrayOfRandomIntegers = new int[theDegree + 1];
        
        ARandomNumberGenerator theRandomNumberGenerator = new ARandomNumberGenerator();
        
        for (int i = 0; i < theDegree + 1; i++) {
        	theArrayOfRandomIntegers[i] =
        		theRandomNumberGenerator.getARandomIntegerInclusivelyBetween(theLowerLimit, theUpperLimit);
        }
        
        return theArrayOfRandomIntegers;
    	
    }
    
}
