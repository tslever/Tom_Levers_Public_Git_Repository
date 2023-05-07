package com.tsl.more_operations_with_arrays.input_output_utilities;


import org.junit.jupiter.api.Test;
import com.tsl.more_operations_with_arrays.AnIntegerOverflowException;
import com.tsl.more_operations_with_arrays.App;


/**
 * AskAboutAndReadALowerLimitAndAnUpperLimitForAnIntegerTest encapsulates JUnit tests of core functionality of the
 * method askAboutAndReadALowerLimitAndAnUpperLimitForAnInteger of class AnInputOutputManager.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
public class AskAboutAndReadALowerLimitAndAnUpperLimitForAnIntegerTest {
	
	
	/**
	 * testAskAboutAndReadALowerLimitAndAnUpperLimitForAnIntegerForValidRange tests
	 * askAboutAndReadALowerLimitAndAnUpperLimitForAnInteger by displaying the main output for a valid range
	 * bounded by a proposed lower limit and a proposed upper limit.
	 */
	@Test
	public void testAskAboutAndReadALowerLimitAndAnUpperLimitForAnIntegerForValidRange() {
		
		System.out.println("Running testAskAboutAndReadALowerLimitAndAnUpperLimitForAnIntegerForValidRange.");
		
		int theProposedLowerLimit = -1073741823;
		int theProposedUpperLimit =  1073741823;
		
		try {		
			if (theProposedLowerLimit > theProposedUpperLimit) {
				throw new ALowerLimitIsGreaterThanUpperLimitException(
					"The proposed lower limit is greater than the proposed upper limit.");
			}
			
			if (theProposedUpperLimit > App.THE_MAXIMUM_INTEGER + theProposedLowerLimit - 1) {
				throw new AnIntegerOverflowException(
					"The proposed range [lower limit, upper limit] is too wide for a random number generator.\n" +
					"A possible range is [-1,073,741,823, 1,073,741,823]."
				);
			}
			
			System.out.println(
				"The proposed lower and upper limits are " +
				theProposedLowerLimit +
				" and " +
				theProposedUpperLimit +
				"."
			);
			
		}
		
		catch (ALowerLimitIsGreaterThanUpperLimitException theLowerLimitIsGreaterThanUpperLimitException) {
			System.out.println(theLowerLimitIsGreaterThanUpperLimitException.getMessage());
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	/**
	 * testAskAboutAndReadALowerLimitAndAnUpperLimitForAnIntegerForLowerLimitHigher tests
	 * askAboutAndReadALowerLimitAndAnUpperLimitForAnInteger by displaying the message of a lower limit is greater than
	 * an upper limit exception thrown when a proposed lower limit is higher than a proposed upper limit.
	 */
	@Test
	public void testAskAboutAndReadALowerLimitAndAnUpperLimitForAnIntegerForLowerLimitHigher() {
		
		System.out.println("Running testAskAboutAndReadALowerLimitAndAnUpperLimitForAnIntegerForLowerLimitHigher.");
		
		int theProposedLowerLimit = 1;
		int theProposedUpperLimit = -1;
		
		try {		
			if (theProposedLowerLimit > theProposedUpperLimit) {
				throw new ALowerLimitIsGreaterThanUpperLimitException(
					"The proposed lower limit is greater than the proposed upper limit.");
			}
			
			if (theProposedUpperLimit > App.THE_MAXIMUM_INTEGER + theProposedLowerLimit - 1) {
				throw new AnIntegerOverflowException(
					"The proposed range [lower limit, upper limit] is too wide for a random number generator.\n" +
					"A possible range is [-1,073,741,823, 1,073,741,823]."
				);
			}
			
			System.out.println(
				"The proposed lower and upper limits are " +
				theProposedLowerLimit +
				" and " +
				theProposedUpperLimit +
				"."
			);
			
		}
		
		catch (ALowerLimitIsGreaterThanUpperLimitException theLowerLimitIsGreaterThanUpperLimitException) {
			System.out.println(theLowerLimitIsGreaterThanUpperLimitException.getMessage());
		}
		
		catch (AnIntegerOverflowException theIntegerOverflowException) {
			System.out.println(theIntegerOverflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
}
