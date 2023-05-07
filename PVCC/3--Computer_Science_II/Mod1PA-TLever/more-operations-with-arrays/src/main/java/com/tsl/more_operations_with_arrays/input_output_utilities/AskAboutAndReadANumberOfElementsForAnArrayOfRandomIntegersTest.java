package com.tsl.more_operations_with_arrays.input_output_utilities;


import org.junit.jupiter.api.Test;


/**
 * AskAboutAndReadANumberOfElementsForAnArrayOfRandomIntegersTest encapsulates JUnit tests of core functionality of the
 * method askAboutAndReadANumberOfElementsForAnArrayOfRandomIntegers of class AnInputOutputManager.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
public class AskAboutAndReadANumberOfElementsForAnArrayOfRandomIntegersTest {

	
	/**
	 * testAskAboutAndReadANumberOfElementsForAnArrayOfRandomIntegersWithAZeroNumberOfElements tests
	 * askAboutAndReadALowerLimitAndAnUpperLimitForAnInteger by displaying the main output for an array of random
	 * integers with a zero number of elements.
	 */
	@Test
	public void testAskAboutAndReadANumberOfElementsForAnArrayOfRandomIntegersWithAZeroNumberOfElements() {
		
		System.out.println(
			"Running testAskAboutAndReadANumberOfElementsForAnArrayOfRandomIntegersWithAZeroNumberOfElements");
		
		try {
			int theProposedNumberOfElements = 0;
			
			if (theProposedNumberOfElements < 0) {
				throw new AProposedNumberOfElementsIsNegativeException("The proposed number of elements is negative.");
			}
			
			System.out.println(theProposedNumberOfElements);
			
		}
		catch (AProposedNumberOfElementsIsNegativeException theReadIntegerIsNegativeException) {
			System.out.println(theReadIntegerIsNegativeException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	/**
	 * testAskAboutAndReadANumberOfElementsForAnArrayOfRandomIntegersWithANegativeNumberOfElements tests
	 * askAboutAndReadALowerLimitAndAnUpperLimitForAnInteger by displaying the message of a a read integer is negative
	 * exception thrown when the proposed number of elements is negative.
	 */
	@Test
	public void testAskAboutAndReadANumberOfElementsForAnArrayOfRandomIntegersWithANegativeNumberOfElements() {
		
		System.out.println(
			"Running testAskAboutAndReadANumberOfElementsForAnArrayOfRandomIntegersWithANegativeNumberOfElements");
		
		try {
			int theProposedNumberOfElements = -1;
			
			if (theProposedNumberOfElements < 0) {
				throw new AProposedNumberOfElementsIsNegativeException("The proposed number of elements is negative.");
			}
			
			System.out.println(theProposedNumberOfElements);
			
		}
		catch (AProposedNumberOfElementsIsNegativeException theReadIntegerIsNegativeException) {
			System.out.println(theReadIntegerIsNegativeException.getMessage());
		}
		
		System.out.println();
		
	}
	
}
