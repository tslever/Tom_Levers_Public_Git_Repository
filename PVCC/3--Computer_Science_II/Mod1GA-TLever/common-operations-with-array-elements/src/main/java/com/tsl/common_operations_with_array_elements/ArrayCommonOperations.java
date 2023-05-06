package com.tsl.common_operations_with_array_elements;


import java.util.Arrays;
import java.util.Random;


/**
* @author EMILIA BUTU
* @version 1.0
* @since   2020-07
*
* Student name:  Tom Lever
* Completion date: 05/17/21
*
* ArrayCommonOperations ecapsulates the entry point to a program to perform common operations on the elements of a
* one-dimensional array.
* The elements of the array are random numbers of type integer, with values between 1 and 100.
* The calculated values are stored, and displayed at the end of the program.
*/

public class ArrayCommonOperations {
	
	
	/**
	 * main represents the entry point of the program.
	 * main performs the common operations on the elements of a one-dimensional array.
	 * @param args
	 */
    public static void main(String [] args) {
    	
    	//*** Task #1: Define and instantiate an array of integer numbers, with 10 elements
        int[] theArrayOfIntegersWithTenElements = new int[10];
        
        
        //*** Task #2: Fill in the array with integer numbers from 1 to 100
        ARandomNumberGenerator theRandomNumberGenerator = new ARandomNumberGenerator();
        
        int i;
        for (i = 0; i < theArrayOfIntegersWithTenElements.length; i++) {
        	theArrayOfIntegersWithTenElements[i] = theRandomNumberGenerator.getARandomIntegerInclusivelyBetween(1, 100);
        }
        
        
		//*** Task #3: 	define another array, named temp, and copy the initial array in it.
		//*				This allows to preserve the original array
        int[] temp = theArrayOfIntegersWithTenElements.clone();
        
        
		//*** Task #4:  Define the variables you need to calculate the following values,
		//*				and initialize them with appropriate values.
        int theSumOfTheIntegersInTheArray = 0;
        int theNumberOfEvenValuesInTheArray = 0;
        int theMinimumValueInTheArray = 0;
        int theMaximumValueInTheArray = 0;
        
        
        //*** Task #5: 	Print the original array
        System.out.println("The original array has the following values:");
        for (i = 0; i < theArrayOfIntegersWithTenElements.length - 1; i++) {
        	System.out.print(theArrayOfIntegersWithTenElements[i] + "\t");
        }
        System.out.println(theArrayOfIntegersWithTenElements[theArrayOfIntegersWithTenElements.length - 1]);
        
        
        //*** Task #6: 	Calculate the sum of all values
        theSumOfTheIntegersInTheArray = Arrays.stream(theArrayOfIntegersWithTenElements).sum();
        
        
        //*** Task #7: 	Count the number of even values
        for (i = 0; i < theArrayOfIntegersWithTenElements.length; i++) {
        	if (MathUtilities.isEven(theArrayOfIntegersWithTenElements[i])) {
        		theNumberOfEvenValuesInTheArray++;
        	}
        }
        
        
        //*** Task #8: 	Calculate the minimum value in the array
        theMinimumValueInTheArray = Arrays.stream(theArrayOfIntegersWithTenElements).min().getAsInt();


        //*** Task #9: 	Calculate the maximum value in the array
        theMaximumValueInTheArray = Arrays.stream(theArrayOfIntegersWithTenElements).max().getAsInt();
        
        
        //*** Task #10: Replace the elements that are divisible by 3, with their value plus 2
        for (i = 0; i < theArrayOfIntegersWithTenElements.length; i++) {
        	if (MathUtilities.isDivisibleByThree(theArrayOfIntegersWithTenElements[i])) {
        		temp[i] += 2;
        	}
        }
        
        
        //*** Task #11: Display the new array after replacement
        System.out.println("The new array after replacement has the following values:");
        for (i = 0; i < temp.length - 1; i++) {
        	System.out.print(temp[i] + "\t");
        }
        System.out.println(temp[temp.length - 1]);
        
        
        //*** Task #12: Display the calculated values.
        System.out.println("Sum of values in the array: " + theSumOfTheIntegersInTheArray);
        System.out.println("Count of even values in the array: " + theNumberOfEvenValuesInTheArray);
        System.out.println("Minimum value in the array: " + theMinimumValueInTheArray);
        System.out.println("Maximum value in the array: " + theMaximumValueInTheArray);
        
    }
    
}


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
	 * ARandomNumberGenerator() is the constructor for ARandomNumberGenerator.
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
    int getARandomIntegerInclusivelyBetween(int theLowerLimit, int theUpperLimit) {
    	
    	return this.thePopularRandomNumberGenerator.nextInt((theUpperLimit - theLowerLimit) + 1) + theLowerLimit;
    	// throws IllegalArgumentException when lower limit is greater than upper limit.
    	
    }
	
}


/**
 * MathUtilities encapsulates methods representing common mathematical operations.
 * @author Tom Lever
 * @version 1.0
 * @since 05/17/21
 *
 */
class MathUtilities {

	/**
	 * isEven indicates whether an integer is even or not.
	 * @param theInteger
	 * @return
	 */
	public static boolean isEven(int theInteger) {
		
		return (theInteger % 2 == 0);
		
	}
	
	
	/**
	 * isDivisibleByThree indicates whether an integer is divisible by 3 or not.
	 * @param theInteger
	 * @return
	 */
	public static boolean isDivisibleByThree(int theInteger) {
		
		return (theInteger % 3 == 0);
		
	}
	
}