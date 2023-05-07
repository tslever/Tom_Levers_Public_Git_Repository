package com.tsl.polynomials;


import java.util.Arrays;


/**
 * Polynomial represents the structure of a polynomial.
 * @author Tom
 *
 */
class Polynomial {
	
	
	/**
	 * degree is an attribute of Polynomial that represents the degree of a polynomial.
	 */
	private int degree;

	
	/**
	 * coefficients is a component of Polynomial and an array of integers representing the coefficients
	 * of the polynomial.
	 * The integer at index 0 corresponds to x^0; the integer at index coefficients.length-1 corresponds to x^{degree}.
	 */
	private int[] coefficients;
	
	
	/**
	 * Polynomial(int theDegreeToUse, int[] theCoefficientsToUse) is the two-argument constructor for Polynomial,
	 * which throws an invalid degree exception if the degree does not have a magnitude that can be expressed as an
	 * integer, otherwise sets the degree to the magnitude of the degree to use, throws an exception if the number of
	 * coefficients to use is not one more than the degree, resets the degree of the polynomial to -1 if the polynomial
	 * is the zero polynomial, and sets the coefficients to the coefficients to use.
	 * @param theDegreeToUse
	 * @param theCoefficientsToUse
	 */
	public Polynomial(int theDegreeToUse, int[] theArrayOfCoefficients)
		throws AnInvalidDegreeException, AnInvalidArrayOfCoefficientsGivenDegreeException {
		
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
		
		if (areAllValuesZeroIn(theArrayOfCoefficients)) {
			this.degree = -1;
			this.coefficients = new int[] {0};
		}
		else {
			this.coefficients = theArrayOfCoefficients;	
		}
		
	}
	
	
	
	/**
	 * areAllValuesZeroIn indicates whether or not are values in an array of coefficients are zero.
	 * @param theArrayOfCoefficients
	 * @return
	 */
	private boolean areAllValuesZeroIn(int[] theArrayOfCoefficients) {
		
		boolean areAllValuesZero = true;
		
		for (int i = 0; i < theArrayOfCoefficients.length; i++) {
			if (theArrayOfCoefficients[i] != 0) {
				areAllValuesZero = false;
				break;
			}
		}
		
		return areAllValuesZero;
		
	}
	
	
	/**
	 * adjustPoly, if the degree of this polynomial is greater than 0 and the coefficient corresponding to the degree
	 * is 0, decrements the degree and removes the above coefficient from the array of coefficients for this polynomial.
	 * If degree is -1, coefficients is {0} and nothing needs to change.
	 * If degree is 0, coefficients is {<some non-zero integer>}, per the constructor of Polynomial.
	 */
	public void adjustPoly() {
		
		if ((this.degree > 0) && (this.coefficients[this.degree] == 0)) {
			degree -= 1;
			this.coefficients = Arrays.copyOfRange(coefficients, 0, this.degree - 1);
		}
		
	}
	
	
	/**
	 * getDegree provides this polynomial's degree.
	 * @return
	 */
	public int getDegree() {
		return this.degree;
	}
	
	
	/**
	 * getCoefficients provides this polynomial's array of coefficients.
	 * @return
	 */
	public int[] getCoefficients() {
		return this.coefficients;
	}
	
	
	/**
	 * checkTheMultiplicationOf throws an integer-overflow exception if multiplication of a first integer and a
	 * second integer would result in a product greater than the maximum integer or less than the minimum integer.
	 * @param theFirstInteger
	 * @param theSecondInteger
	 * @throws AnIntegerOverflowException
	 */
	private void checkTheMultiplicationOf(int theFirstInteger, int theSecondInteger) throws AnIntegerOverflowException {
		
		if (((theFirstInteger > 0) && (theSecondInteger > 0) && (theFirstInteger > PolynomialDriver.THE_MAXIMUM_INTEGER / theSecondInteger)) ||
			((theFirstInteger < 0) && (theSecondInteger > 0) && (theFirstInteger < PolynomialDriver.THE_MINIMUM_INTEGER / theSecondInteger)) ||
			((theFirstInteger > 0) && (theSecondInteger < 0) && (theFirstInteger > PolynomialDriver.THE_MINIMUM_INTEGER / theSecondInteger)) ||
			((theFirstInteger < 0) && (theSecondInteger < 0) && (theFirstInteger < PolynomialDriver.THE_MAXIMUM_INTEGER / theSecondInteger))) {
			
			throw new AnIntegerOverflowException(
				"Integer-overflow exception: The product of " +
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
		
	}
	
	
	/**
	 * exponentiate provides the product of exponentiating a base by a power.
	 * @param theBase
	 * @param thePower
	 * @return
	 * @throws ANotSufficientlyImplementedException
	 * @throws AnIntegerOverflowException
	 */
	private int exponentiate(int theBase, int thePower)
		throws ANotSufficientlyImplementedException, AnIntegerOverflowException {
		
		if (thePower < 0) {
			throw new ANotSufficientlyImplementedException(
				"Exponentiation with negative integer powers has not been implemented yet.");
		}
		
		if (thePower == 0) {
			return 1;
		}
		
		if (thePower == 1) {
			return theBase;
		}
		
		int theProduct = theBase;
		
		for (int i = 2; i <= thePower; i++) {
			
			checkTheMultiplicationOf(theBase, theProduct);
			
			theProduct *= theBase;
			
		}
		
		return theProduct;
		
	}
	
	
	/**
	 * checkTheAdditionOf throws an integer-overflow exception if addition of a first integer and a
	 * second integer would result in a sum greater than the maximum integer or less than the minimum integer.
	 * @param theFirstInteger
	 * @param theSecondInteger
	 * @throws AnIntegerOverflowException
	 */
	private void checkTheAdditionOf(int theFirstInteger, int theSecondInteger) throws AnIntegerOverflowException {
		
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
		
	}
	
	
	/**
	 * multiply provides the product of multiplying a first integer by a second integer.
	 * @param theFirstInteger
	 * @param theSecondInteger
	 * @return
	 * @throws AnIntegerOverflowException
	 */
	private int multiply(int theFirstInteger, int theSecondInteger) throws AnIntegerOverflowException {
		
		checkTheMultiplicationOf(theFirstInteger, theSecondInteger);
		
		return theFirstInteger * theSecondInteger;
		
	}
	
	
	/**
	 * evaluate provides the scalar output of this polynomial given input x.
	 * @param x
	 */
	public int evaluate(int x) throws ANotSufficientlyImplementedException, AnIntegerOverflowException {
		
		int theScalarOutput = 0;
		
		int thePresentTerm;
		for (int i = 0; i < this.coefficients.length; i++) {
			
			thePresentTerm = multiply(this.coefficients[i], exponentiate(x, i));
			
			checkTheAdditionOf(thePresentTerm, theScalarOutput);
			
			theScalarOutput += thePresentTerm;
		}
		
		return theScalarOutput;
		
	}
	
	
	/**
	 * addPoly provides the algebraic sum of this polynomial and polynomial p.
	 * @param p
	 * @return
	 */
	public Polynomial addPoly(Polynomial p)
		throws AnIntegerOverflowException, AnInvalidDegreeException, AnInvalidArrayOfCoefficientsGivenDegreeException {
		
		Polynomial thePolynomialWithTheLargerDegree = (this.degree > p.degree) ? this : p; 
		
		Polynomial thePolynomialWithTheSmallerDegree = (this.degree > p.degree) ? p : this; 
		
		int[] theArrayOfCoefficientsForTheSum = thePolynomialWithTheLargerDegree.coefficients.clone();
		
		for (int i = 0; i < thePolynomialWithTheSmallerDegree.coefficients.length; i++) {
			
			checkTheAdditionOf(theArrayOfCoefficientsForTheSum[i], thePolynomialWithTheSmallerDegree.coefficients[i]);
			
			theArrayOfCoefficientsForTheSum[i] += thePolynomialWithTheSmallerDegree.coefficients[i];
		}
		
		return new Polynomial(thePolynomialWithTheLargerDegree.degree, theArrayOfCoefficientsForTheSum);
		
	}
	
	
	/**
	 * subtractPoly provides the algebraic difference produced by taking polynomial p from this polynomial.
	 * @param p
	 * @return
	 * @throws AnIntegerOverflowException
	 * @throws AnInvalidDegreeException
	 * @throws AnInvalidArrayOfCoefficientsGivenDegreeException
	 */
	public Polynomial subtractPoly(Polynomial p)
		throws AnIntegerOverflowException, AnInvalidDegreeException, AnInvalidArrayOfCoefficientsGivenDegreeException {
		
		int[] theCoefficientsOfThePolynomialOppositeP = p.coefficients.clone();
		
		for (int i = 0; i < theCoefficientsOfThePolynomialOppositeP.length; i++) {
			theCoefficientsOfThePolynomialOppositeP[i] = multiply(p.coefficients[i], -1);
		}
		
		return addPoly(new Polynomial(p.degree, theCoefficientsOfThePolynomialOppositeP));
		
	}
	
	
	/**
	 * derivative provides a polynomial that represents the algebraic derivative of this polynomial.
	 * @return
	 * @throws AnIntegerOverflowException
	 * @throws AnInvalidDegreeException
	 * @throws AnInvalidArrayOfCoefficientsGivenDegreeException
	 */
	public Polynomial derivative()
		throws AnIntegerOverflowException, AnInvalidDegreeException, AnInvalidArrayOfCoefficientsGivenDegreeException {
		
		int[] theCoefficientsOfTheDerivative = Arrays.copyOfRange(this.coefficients, 1, this.coefficients.length);
		
		for (int i = 0; i < theCoefficientsOfTheDerivative.length; i++) {
			theCoefficientsOfTheDerivative[i] = multiply(theCoefficientsOfTheDerivative[i], i+1);
		}
		
		return new Polynomial(this.degree - 1, theCoefficientsOfTheDerivative);
		
	}
	
	
	/**
	 * equals indicates whether or not this polynomial and polynomial p are equal in degree and coefficients.
	 * @param p
	 * @return
	 */
	public boolean equals(Polynomial p) {
		
		return ((this.degree == p.degree) && (Arrays.equals(this.coefficients, p.coefficients)));
		
	}
	
	
	/**
	 * toString provides a standard mathematical representation of this polynomial, without variable and assignment.
	 */
	@Override
	public String toString() {
		
		String theOutputString = "";
		
		if (this.degree > 1) {
			if (this.coefficients[this.degree] > 1) {
				theOutputString += this.coefficients[this.degree] + "x^" + this.degree;
			}
			else if (this.coefficients[this.degree] == 1) {
				theOutputString += "x^" + this.degree;
			}
			else if (this.coefficients[this.degree] == -1) {
				theOutputString += "-x^" + this.degree;
			}
			else if (this.coefficients[this.degree] < -1) {
				theOutputString += this.coefficients[this.degree] + "x^" + this.degree;
			}
		}
		
		for (int i = this.degree - 1; i > 1; i--) {
			
			if (this.coefficients[i] > 1) {
				theOutputString += " + " + this.coefficients[i] + "x^" + i;
			}
			else if (this.coefficients[i] == 1) {
				theOutputString += " + x^" + i;
			}
			else if (this.coefficients[i] == -1) {
				theOutputString += " - x^" + i;
			}
			else if (this.coefficients[i] < -1) {
				theOutputString += " - " + -this.coefficients[i] + "x^" + i;
			}
			
		}
		
		if (this.degree > 0) {
			if (this.coefficients[1] > 1) {
				theOutputString += " + " + this.coefficients[1] + "x";
			}
			else if (this.coefficients[1] == 1) {
				theOutputString += " + x";
			}
			else if (this.coefficients[1] == -1) {
				theOutputString += " - x";
			}
			else if (this.coefficients[1] < -1) {
				theOutputString += " - " + -this.coefficients[1] + "x";
			}
		}
		
		if (this.coefficients[0] > 0) {
			theOutputString += " + " + this.coefficients[0];
		}
		else if (this.coefficients[0] < 0) {
			theOutputString += " - " + -this.coefficients[0];
		}
		else if (this.coefficients.length == 1) {
			theOutputString += "0";
		}
		
		return theOutputString;
		
	}
	
}
