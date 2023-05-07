package com.tsl.polynomials;


import java.util.Arrays;
import org.junit.jupiter.api.Test;


/**
 * PolynomialTest encapsulates JUnit tests of core functionality of the constructor
 * Polynomial(int theDegreeToUse, int[] theArrayOfCoefficients) of class Polynomial.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
public class PolynomialTest {

	
	/**
	 * degree is an attribute of PolynomialTest as degree is an attribute of Polynomial.
	 */
	private int degree;
	
	
	/**
	 * testPolynomialAndNegativeProposedDegree tests Polynomial(int theDegreeToUse, int[] theArrayOfCoefficients)
	 * for a valid negative proposed degree.
	 */
	@Test
	public void testPolynomialAndValidNegativeProposedDegree() {
	
		System.out.println("Running testPolynomialAndValidNegativeProposedDegree.");
		
		int theDegreeToUse = PolynomialDriver.THE_MINIMUM_INTEGER + 1;
		
		try {
		
			if (theDegreeToUse == PolynomialDriver.THE_MINIMUM_INTEGER) {
				throw new AnInvalidDegreeException("Degree must be larger than " + PolynomialDriver.THE_MINIMUM_INTEGER);
			}
			
			if (theDegreeToUse < 0) {
				System.out.println(
					"Warning: Proposed degree " +
					theDegreeToUse +
					" is negative; making the polynomial degree equal to the magnitude of the proposed degree."
				);
				this.degree = -theDegreeToUse;
				System.out.println(
					"The proposed degree is " + theDegreeToUse + "; the stored degree is " + this.degree + ".");
			}
			else {
				this.degree = theDegreeToUse;
				System.out.println(
					"The proposed degree is " + theDegreeToUse + "; the stored degree is " + this.degree + ".");
			}
		
		}
		
		catch (AnInvalidDegreeException theInvalidDegreeException) {
			System.out.println(theInvalidDegreeException.getMessage());
		}
		
		System.out.println();
	
	}
	
	
	/**
	 * testPolynomialAndPositiveProposedDegree tests Polynomial(int theDegreeToUse, int[] theArrayOfCoefficients)
	 * for a valid positive proposed degree.
	 */
	@Test
	public void testPolynomialAndValidPositiveProposedDegree() {
	
		System.out.println("Running testPolynomialAndValidPositiveProposedDegree.");
		
		int theDegreeToUse = PolynomialDriver.THE_MAXIMUM_INTEGER;
		
		try {
		
			if (theDegreeToUse == PolynomialDriver.THE_MINIMUM_INTEGER) {
				throw new AnInvalidDegreeException("Degree must be larger than " + PolynomialDriver.THE_MINIMUM_INTEGER);
			}
			
			if (theDegreeToUse < 0) {
				System.out.println(
					"Warning: Proposed degree " +
					theDegreeToUse +
					" is negative; making the polynomial degree equal to the magnitude of the proposed degree."
				);
				this.degree = -theDegreeToUse;
				System.out.println(
					"The proposed degree is " + theDegreeToUse + "; the stored degree is " + this.degree + ".");
			}
			else {
				this.degree = theDegreeToUse;
				System.out.println(
					"The proposed degree is " + theDegreeToUse + "; the stored degree is " + this.degree + ".");
			}
		
		}
		
		catch (AnInvalidDegreeException theInvalidDegreeException) {
			System.out.println(theInvalidDegreeException.getMessage());
		}
		
		System.out.println();
	
	}
	
	
	/**
	 * testPolynomialAndNegativeProposedDegree tests Polynomial(int theDegreeToUse, int[] theArrayOfCoefficients)
	 * for an invalid proposed degree.
	 */
	@Test
	public void testPolynomialAndAnInvalidProposedDegree() {
	
		System.out.println("Running testPolynomialAndAnInvalidProposedDegree.");
		
		int theDegreeToUse = PolynomialDriver.THE_MINIMUM_INTEGER;
		
		try {
		
			if (theDegreeToUse == PolynomialDriver.THE_MINIMUM_INTEGER) {
				throw new AnInvalidDegreeException("Degree must be larger than " + PolynomialDriver.THE_MINIMUM_INTEGER);
			}
		
		}
		
		catch (AnInvalidDegreeException theInvalidDegreeException) {
			System.out.println(theInvalidDegreeException.getMessage());
		}
		
		System.out.println();
	
	}
	
	
	/**
	 * testPolynomialAndValidCoefficientsGivenDegree tests Polynomial(int theDegreeToUse, int[] theArrayOfCoefficients)
	 * for a valid array of coefficients given a valid degree.
	 */
	@Test
	public void testPolynomialAndValidCoefficientsGivenValidDegree() {
	
		System.out.println("Running testPolynomialAndValidCoefficientsGivenValidDegree.");
		
		int theDegreeToUse = -1;
		
		int[] theArrayOfCoefficients = new int[] {1, 2};
		
		try {
		
			if (theDegreeToUse == PolynomialDriver.THE_MINIMUM_INTEGER) {
				throw new AnInvalidDegreeException("Degree must be larger than " + PolynomialDriver.THE_MINIMUM_INTEGER);
			}
			
			if (theDegreeToUse < 0) {
				System.out.println(
					"Warning: Proposed degree " +
					theDegreeToUse +
					" is negative; making the polynomial degree equal to the magnitude of the proposed degree."
				);
				this.degree = -theDegreeToUse;
			}
			else {
				this.degree = theDegreeToUse;
			}
			
			if (theArrayOfCoefficients.length != this.degree + 1) {
				throw new AnInvalidArrayOfCoefficientsGivenDegreeException(
					"Given degree " +
					this.degree +
					", the array of coefficients must have " +
					(this.degree + 1) +
					" element(s)."
				);
			}
			
			System.out.println(
				"Valid coefficients given proposed degree " +
				theDegreeToUse +
				": " +
				Arrays.toString(theArrayOfCoefficients)
			);
		
		}
		
		catch (AnInvalidDegreeException theInvalidDegreeException) {
			System.out.println(theInvalidDegreeException.getMessage());
		}
		
		catch (AnInvalidArrayOfCoefficientsGivenDegreeException theInvalidArrayOfCoefficientsGivenDegreeException) {
			System.out.println(theInvalidArrayOfCoefficientsGivenDegreeException.getMessage());
		}
		
		System.out.println();
	
	}
	
	
	/**
	 * testPolynomialAndValidCoefficientsGivenDegree tests Polynomial(int theDegreeToUse, int[] theArrayOfCoefficients)
	 * for an invalid array of coefficients given a valid degree.
	 */
	@Test
	public void testPolynomialAndInvalidCoefficientsGivenValidDegree() {
	
		System.out.println("Running testPolynomialAndInvalidCoefficientsGivenValidDegree.");
		
		int theDegreeToUse = -1;
		
		int[] theArrayOfCoefficients = new int[] {1, 2, 3};
		
		try {
		
			if (theDegreeToUse == PolynomialDriver.THE_MINIMUM_INTEGER) {
				throw new AnInvalidDegreeException("Degree must be larger than " + PolynomialDriver.THE_MINIMUM_INTEGER);
			}
			
			if (theDegreeToUse < 0) {
				System.out.println(
					"Warning: Proposed degree " +
					theDegreeToUse +
					" is negative; making the polynomial degree equal to the magnitude of the proposed degree."
				);
				this.degree = -theDegreeToUse;
			}
			else {
				this.degree = theDegreeToUse;
			}
			
			if (theArrayOfCoefficients.length != this.degree + 1) {
				throw new AnInvalidArrayOfCoefficientsGivenDegreeException(
					"Given degree " +
					this.degree +
					", the array of coefficients must have " +
					(this.degree + 1) +
					" element(s)."
				);
			}
			
			System.out.println(
				"Invalid coefficients given proposed degree " +
				theDegreeToUse +
				": " +
				Arrays.toString(theArrayOfCoefficients)
			);
		
		}
		
		catch (AnInvalidDegreeException theInvalidDegreeException) {
			System.out.println(theInvalidDegreeException.getMessage());
		}
		
		catch (AnInvalidArrayOfCoefficientsGivenDegreeException theInvalidArrayOfCoefficientsGivenDegreeException) {
			System.out.println(theInvalidArrayOfCoefficientsGivenDegreeException.getMessage());
		}
		
		System.out.println();
	
	}
	
	
}
