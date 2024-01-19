package com.tsl.more_operations_with_arrays;


import com.tsl.more_operations_with_arrays.input_output_utilities.AnInputOutputManager;
import java.util.Arrays;
import org.apache.commons.lang3.ArrayUtils;


/**
 * App encapsulates the entry point to a program to perform operations with arrays beyond those in Mod1GA-TLever.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 */
public class App 
{
	public static final int THE_MINIMUM_INTEGER = -2147483648; // TODO: Change public to private upon completion of testing.
	public static final int THE_MAXIMUM_INTEGER = 2147483647;
	
	
	/**
	 * main represents the entry point of the program. main:
	 * - Creates and displays an array of random integers based on user input;
	 * - Calculates and displays the average of the values in the array;
	 * - Calculates and displays the sum of the odd values in the array (or displays an error message);
	 * - Finds and displays the second maximum value in the array (or displays and error message);
	 * - Calculates and displays an array of the squares of the values in the original array (or displays an error
	 *   message); and
	 * - Calculates and displays the number of peaks in the original array.
	 * 
	 * @param args
	 */
    public static void main( String[] args )
    {
    	int[] theArrayOfRandomIntegers = createAnArrayOfRandomIntegers();
        
        System.out.println("The original array has the following values:");
        display(theArrayOfRandomIntegers);
        
        double theAverageOfTheValuesInTheArray = averageTheValuesIn(theArrayOfRandomIntegers);
        System.out.println("The average of the values in the array is: " + theAverageOfTheValuesInTheArray);
        
        try {
        	int theSumOfTheOddValuesInTheArray = sumTheOddValuesIn(theArrayOfRandomIntegers);
        	System.out.println("The sum of odd values in the array is: " + theSumOfTheOddValuesInTheArray);
        }
        catch (AnIntegerOverflowException theIntegerOverflowException) {
        	System.out.println(theIntegerOverflowException.getMessage());
        }
        catch (ASumDoesNotExistException theSumDoesNotExistException) {
        	System.out.println(theSumDoesNotExistException.getMessage());
        }
        
        try {
        	double theSecondMaximumValueInTheArray = findTheSecondMaximumValueIn(theArrayOfRandomIntegers);
        	System.out.println("The second maximum value in the array is: " + theSecondMaximumValueInTheArray);
        }
        catch (AMaximumValueDoesNotExistException theMaximumValueDoesNotExistException) {
        	System.out.println(theMaximumValueDoesNotExistException.getMessage());
        }
        catch (ASecondMaximumValueDoesNotExistException theSecondMaximumValueDoesNotExistException) {
        	System.out.println(theSecondMaximumValueDoesNotExistException.getMessage());
        }
        
        try {
        	int[] theArrayOfSquares = calculateTheArrayOfTheSquaresOfTheValuesIn(theArrayOfRandomIntegers);
        	System.out.println("\nThe array containing squares of the values in the original array has the following values:");
        	display(theArrayOfSquares);
        }
        catch (AnIntegerOverflowException theIntegerOverflowException) {
        	System.out.println(theIntegerOverflowException.getMessage());
        }
        
    	int theNumberOfPeaks = calculateTheNumberOfPeaksIn(theArrayOfRandomIntegers);
    	System.out.println("The number of peaks in the array: " + theNumberOfPeaks);
    }
    
    
    /**
     * calculateTheNumberOfPeaksIn calculates the number of peaks in the array of random integers.
     * @param theArrayOfRandomIntegers
     * @return
     */
    private static int calculateTheNumberOfPeaksIn(int[] theArrayOfRandomIntegers) {
    	
    	int theNumberOfPeaks = 0;
    	
    	for (int i = 1; i < theArrayOfRandomIntegers.length - 1; i++) {
    		
    		if ((theArrayOfRandomIntegers[i] > theArrayOfRandomIntegers[i-1]) &&
    			(theArrayOfRandomIntegers[i] > theArrayOfRandomIntegers[i+1])) {
    			
    			theNumberOfPeaks++;
    			
    		}
    		
    	}
    	
    	return theNumberOfPeaks;
    	
    }
    
    
    /**
     * adjust adjusts a number of summands and and a sum based on an element (or throws an integer-overflow exception).
     * @param theNumberOfSummands
     * @param theSum
     * @param theElement
     * @return
     * @throws AnIntegerOverflowException
     */
    private static int[] adjust(int theNumberOfSummands, int theSum, int theElement) throws AnIntegerOverflowException {
    	
		theNumberOfSummands++;
		
		if ((((theSum >= 0) && (theElement > 0)) && (theElement > (THE_MAXIMUM_INTEGER - theSum)))) {
			throw new AnIntegerOverflowException(
				"Because of " + theElement +
				", the sum of the odd values in the array is greater than the maximum integer."
			);
		}
		
		if ((((theSum < 0) && (theElement < 0)) && (theElement < (THE_MINIMUM_INTEGER - theSum)))) {
			throw new AnIntegerOverflowException(
				"Because of " + theElement +
				", the sum of the odd values in the array is less than the minimum integer."
			);
		}
		
		theSum += theElement;
		
		return new int[] {theNumberOfSummands, theSum};
    	
    }
    
    
    /**
     * arrangeInAnOscillatoryPattern arranges an array of random integers in an oscillatory pattern that looks like
     * a damping sinusoid that starts high, goes low, and fades away.
     * @param theArrayOfRandomIntegers
     * @return
     */
    private static int[] arrangeInAnOscillatoryPattern(int[] theArrayOfRandomIntegers) {
    	
    	int[] theNumbersOfNonNegativeAndNegativeIntegersInTheArray =
    		getTheNumbersOfNonNegativeAndNegativeIntegersIn(theArrayOfRandomIntegers);
    	int theNumberOfNonNegativeIntegers = theNumbersOfNonNegativeAndNegativeIntegersInTheArray[0];
    	int theNumberOfNegativeIntegers = theNumbersOfNonNegativeAndNegativeIntegersInTheArray[1];
    	
    	int[][] theArraysOfNonNegativeAndNegativeIntegersInDescendingOrder =
    		createArraysOfNonNegativeAndNegativeIntegersInDescendingOrderBasedOn(
    			theArrayOfRandomIntegers, theNumberOfNonNegativeIntegers, theNumberOfNegativeIntegers);
    	int[] theArrayOfNonNegativeIntegersInDescendingOrder =
    		theArraysOfNonNegativeAndNegativeIntegersInDescendingOrder[0];
    	int[] theArrayOfNegativeIntegersInDescendingOrder =
        	theArraysOfNonNegativeAndNegativeIntegersInDescendingOrder[1];

    	int[] theArrayInAnOscillatoryPattern =
    		createAnArrayInAnOscillatoryPatternBasedOn(
    			theArrayOfNonNegativeIntegersInDescendingOrder,
    			theArrayOfNegativeIntegersInDescendingOrder,
    			theNumberOfNonNegativeIntegers,
    			theNumberOfNegativeIntegers
    		);
    	
    	return theArrayInAnOscillatoryPattern;
    	
    }
    
    
    /**
     * averageTheValuesIn averages the values in an array of random integers.
     * @param theArrayOfRandomIntegers
     * @return
     */
    private static double averageTheValuesIn(int[] theArrayOfRandomIntegers) {
    	return Arrays.stream(theArrayOfRandomIntegers).average().getAsDouble();
    }
    
    
    /**
     * calculateTheArrayOfTheSquaresOfTheValuesIn calculates an array of the squares of the values in an array of
     * random integers (or throws an integer-overflow exception).
     * @param theArrayOfRandomIntegers
     * @return
     * @throws AnIntegerOverflowException
     */
    // TODO: Change public to private upon completion of testing.
    public static int[] calculateTheArrayOfTheSquaresOfTheValuesIn(int[] theArrayOfRandomIntegers)
    	throws AnIntegerOverflowException {
    	
    	int[] theArrayOfSquares = new int[theArrayOfRandomIntegers.length];
    	
    	int theElement;
    	for (int i = 0; i < theArrayOfRandomIntegers.length; i++) {
    		
    		theElement = theArrayOfRandomIntegers[i];
    		
    		if (((theElement > 0) && (theElement > THE_MAXIMUM_INTEGER / theElement)) ||
    			((theElement < 0) && (theElement < THE_MAXIMUM_INTEGER / theElement))) {
    			throw new AnIntegerOverflowException(
    				"The magnitude of " + theElement + " is too large for the element to be squared.\n" +
    				"Try restricting the range of values to [-46,340, 46,340].");
    		}
    		
    		theArrayOfSquares[i] = theElement * theElement;
    		
    	}
    	
    	return theArrayOfSquares;
    	
    }
    
    
    /**
     * createAnArrayOfRandomIntegers creates an array of random integers based on requests for parameters from a user.
     * @return
     */
    private static int[] createAnArrayOfRandomIntegers() {
    	
        AnInputOutputManager theInputOutputManager = new AnInputOutputManager();
        
        int theNumberOfElementsInAnArrayOfRandomIntegers =
        	theInputOutputManager.askAboutAndReadANumberOfElementsForAnArrayOfRandomIntegers();
    	
        int[] theArrayOfRandomIntegers = new int[theNumberOfElementsInAnArrayOfRandomIntegers];
        
        ARandomNumberGenerator theRandomNumberGenerator = new ARandomNumberGenerator();
        
        while (true) {
        
	        try {
		        
	            int[] theLowerLimitAndTheUpperLimitForAnInteger =
	                	theInputOutputManager.askAboutAndReadALowerLimitAndAnUpperLimitForAnInteger();
		        
		        for (int i = 0; i < theArrayOfRandomIntegers.length; i++) {
		        	theArrayOfRandomIntegers[i] =
		        		theRandomNumberGenerator.getARandomIntegerInclusivelyBetween(
		        			theLowerLimitAndTheUpperLimitForAnInteger[0], theLowerLimitAndTheUpperLimitForAnInteger[1]);
		        }
		        
		        return theArrayOfRandomIntegers;
	        
	        }
	        
	        catch (AnIntegerOverflowException theIntegerOverflowException) {
	        	
	        	theIntegerOverflowException.getMessage();
	        	
	        }
        
        }        
    	
    }
    
    
    /**
     * createAnArrayInAnOscillatoryPatternBasedOn creates an array in an oscillator pattern based on an array of
     * non-negative integers in descending order from an original array of random integers, an array of negative
     * integers in descending order, and integer attributes of those arrays.
     * @param theArrayOfNonNegativeIntegersInDescendingOrder
     * @param theArrayOfNegativeIntegersInDescendingOrder
     * @param theNumberOfNonNegativeIntegers
     * @param theNumberOfNegativeIntegers
     * @return
     */
    private static int[] createAnArrayInAnOscillatoryPatternBasedOn(
    	int[] theArrayOfNonNegativeIntegersInDescendingOrder,
    	int[] theArrayOfNegativeIntegersInDescendingOrder,
    	int theNumberOfNonNegativeIntegers,
    	int theNumberOfNegativeIntegers
    ) {
    	
    	int[] theArrayInAnOscillatoryPattern = new int[theNumberOfNonNegativeIntegers + theNumberOfNegativeIntegers];

    	int theLowerNumber =
    		(theNumberOfNonNegativeIntegers < theNumberOfNegativeIntegers) ?
    			theNumberOfNonNegativeIntegers : theNumberOfNegativeIntegers;
    	
    	int i;
    	for (i = 0; i < theLowerNumber; i++) {
    		theArrayInAnOscillatoryPattern[2*i] = theArrayOfNonNegativeIntegersInDescendingOrder[i];
    		theArrayInAnOscillatoryPattern[2*i+1] = theArrayOfNegativeIntegersInDescendingOrder[i];
    	}
    	
    	int[] theLargerArray =
			(theNumberOfNegativeIntegers > theNumberOfNonNegativeIntegers) ?
				theArrayOfNegativeIntegersInDescendingOrder : theArrayOfNonNegativeIntegersInDescendingOrder;
    	
    	for (i = 0; i < theLargerArray.length - theLowerNumber; i++) {
    		theArrayInAnOscillatoryPattern[2*theLowerNumber + i] = theLargerArray[theLowerNumber + i];
    	}
    	
    	return theArrayInAnOscillatoryPattern;
    	
    }
    
    
    /**
     * createArraysOfNonNegativeAndNegativeIntegersInDescendingOrderBasedOn creates arrays of non-negative and
     * negative integers in descending based on an array of random integers.
     * @param theArrayOfRandomIntegers
     * @param theNumberOfNonNegativeIntegers
     * @param theNumberOfNegativeIntegers
     * @return
     */
    private static int[][] createArraysOfNonNegativeAndNegativeIntegersInDescendingOrderBasedOn(
    	int[] theArrayOfRandomIntegers, int theNumberOfNonNegativeIntegers, int theNumberOfNegativeIntegers) {
    	
    	int[] theArrayOfNonNegativeIntegers = new int[theNumberOfNonNegativeIntegers];
    	int[] theArrayOfNegativeIntegers = new int[theNumberOfNegativeIntegers];
    	
    	int thePresentIndexInTheArrayOfNonNegativeIntegers = 0;
    	int thePresentIndexInTheArrayOfNegativeIntegers = 0;
    	
    	for (int i = 0; i < theArrayOfRandomIntegers.length; i++) {
    		if (theArrayOfRandomIntegers[i] >= 0) {
    			theArrayOfNonNegativeIntegers[thePresentIndexInTheArrayOfNonNegativeIntegers] =
    				theArrayOfRandomIntegers[i];
    			thePresentIndexInTheArrayOfNonNegativeIntegers++;
    		}
    		else {
    			theArrayOfNegativeIntegers[thePresentIndexInTheArrayOfNegativeIntegers] = theArrayOfRandomIntegers[i];
    			thePresentIndexInTheArrayOfNegativeIntegers++;
    		}
    	}
    	
    	Arrays.sort(theArrayOfNonNegativeIntegers);
    	ArrayUtils.reverse(theArrayOfNonNegativeIntegers);
    	
    	Arrays.sort(theArrayOfNegativeIntegers);
    	
    	return new int[][] {theArrayOfNonNegativeIntegers, theArrayOfNegativeIntegers};
    	
    }
    
    
    /**
     * display displays prettily the values in an array.
     * @param theArrayOfRandomIntegers
     */
    // TODO: Change public to private upon completion of testing.
    public static void display(int[] theArray) {
    	
        for (int i = 0; i < theArray.length - 1; i++) {
        	System.out.print(theArray[i] + "  ");
        }
        System.out.println(theArray[theArray.length - 1] + "\n");
    	
    }
    
    
    /**
     * findTheIndexForTheMaximumValueIn finds the index for the maximum value in an array of random integers (or throws
     * a maximum value does not exist exception).
     * @param theArrayOfRandomIntegers
     * @return
     * @throws AMaximumValueDoesNotExistException
     */
    // TODO: Change public to private upon completion of testing.
    public static int findTheIndexForTheMaximumValueIn(int[] theArrayOfRandomIntegers)
    	throws AMaximumValueDoesNotExistException {
    	
    	if (theArrayOfRandomIntegers.length == 0) {
    		throw new AMaximumValueDoesNotExistException("A maximum value does not exist.");
    	}
    	
    	int theMaximumValue = THE_MINIMUM_INTEGER;
    	int theIndexOfTheMaximumValue = -1;
    	for (int i = 0; i < theArrayOfRandomIntegers.length; i++) {
    		if (theArrayOfRandomIntegers[i] > theMaximumValue) {
    			theMaximumValue = theArrayOfRandomIntegers[i];
    			theIndexOfTheMaximumValue = i;
    		}
    	}
    	
    	return theIndexOfTheMaximumValue;
    	
    }
    
    
    /**
     * findTheSecondMaximumValueIn finds the second maximum value given the array of random integers.
     * @param theArrayOfRandomIntegers
     * @return
     * @throws AMaximumValueDoesNotExistException
     * @throws ASecondMaximumValueDoesNotExistException
     */
    private static int findTheSecondMaximumValueIn(int[] theArrayOfRandomIntegers)
    	throws AMaximumValueDoesNotExistException, ASecondMaximumValueDoesNotExistException {
    	
    	int theIndexOfTheMaximumValue = findTheIndexForTheMaximumValueIn(theArrayOfRandomIntegers);
  
    	return findTheSecondMaximumValueGiven(theArrayOfRandomIntegers, theIndexOfTheMaximumValue);
    	
    }
    
    
    /**
     * findTheSecondMaximumValueGiven finds the second maximum value given an array of random integers and the index
     * of the maximum value (or throws a second maximum value does not exist exception).
     * @param theArrayOfRandomIntegers
     * @param theIndexOfTheMaximumValue
     * @return
     * @throws ASecondMaximumValueDoesNotExistException
     */
    // TODO: Change public to private upon completion of testing.
    public static int findTheSecondMaximumValueGiven(
    	int[] theArrayOfRandomIntegers, int theIndexOfTheMaximumValue)
    	throws ASecondMaximumValueDoesNotExistException {
    	
    	if (theArrayOfRandomIntegers.length < 2) {
    		throw new ASecondMaximumValueDoesNotExistException("A second maximum value does not exist.");
    	}
    	
    	int theSecondMaximumValue = THE_MINIMUM_INTEGER;
    	for (int i = 0; i < theIndexOfTheMaximumValue; i++) {
    		if (theArrayOfRandomIntegers[i] > theSecondMaximumValue) {
    			theSecondMaximumValue = theArrayOfRandomIntegers[i];
    		}
    	}
    	for (int i = theIndexOfTheMaximumValue + 1; i < theArrayOfRandomIntegers.length; i++) {
    		if (theArrayOfRandomIntegers[i] > theSecondMaximumValue) {
    			theSecondMaximumValue = theArrayOfRandomIntegers[i];
    		}
    	}
    	
    	return theSecondMaximumValue;
    	
    }
    
    
    /**
     * getTheNumbersOfNonNegativeAndNegativeIntegersIn gets the numbers of non-negative and negative integers in
     * an array of random integers.
     * @param theArrayOfRandomIntegers
     * @return
     */
    private static int[] getTheNumbersOfNonNegativeAndNegativeIntegersIn(int[] theArrayOfRandomIntegers) {
    	
    	int theNumberOfNonNegativeIntegers = 0;
    	int theNumberOfNegativeIntegers = 0;
    	
    	for (int i = 0; i < theArrayOfRandomIntegers.length; i++) {
    		if (theArrayOfRandomIntegers[i] >= 0) {
    			theNumberOfNonNegativeIntegers++;
    		}
    		else {
    			theNumberOfNegativeIntegers++;
    		}
    	}
    	
    	return new int[] {theNumberOfNonNegativeIntegers, theNumberOfNegativeIntegers};
    	
    }
    
    
    /**
     * sumTheOddValuesIn sums the odd values in an array of random integers (or throws a sum does not exist exception).
     * @param theArrayOfRandomIntegers
     * @return
     * @throws AnIntegerOverflowException
     * @throws ASumDoesNotExistException
     */
    // TODO: Change public to private upon completion of testing.
    public static int sumTheOddValuesIn(int[] theArrayOfRandomIntegers)
    	throws AnIntegerOverflowException, ASumDoesNotExistException {
    	
        int[] theArrayInAnOscillatoryPattern = arrangeInAnOscillatoryPattern(theArrayOfRandomIntegers);
    	
    	int theSumOfTheOddValuesInTheArray = 0;
    	int theNumberOfSummands = 0;
    	
    	int theElement;
    	for (int i = 0; i < theArrayInAnOscillatoryPattern.length; i++) {
    		
    		theElement = theArrayInAnOscillatoryPattern[i];
    		
    		if (MathUtilities.isOdd(theElement)) {
    			
    			int[] theNumberOfSummandsAndTheSum =
    				adjust(theNumberOfSummands, theSumOfTheOddValuesInTheArray, theElement);
    			theNumberOfSummands = theNumberOfSummandsAndTheSum[0];
    			theSumOfTheOddValuesInTheArray = theNumberOfSummandsAndTheSum[1];
    			
    		}
    		
    	}
    	
    	if (theNumberOfSummands == 0) {
    		throw new ASumDoesNotExistException("A sum of the odd values in the array does not exist.");
    	}
    	
    	return theSumOfTheOddValuesInTheArray;
    	
    }
    
}
