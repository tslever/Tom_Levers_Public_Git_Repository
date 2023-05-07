package Com.TSL.SelectionSortUtilities;


import java.util.Arrays;
import org.apache.commons.math3.random.RandomDataGenerator;


/** *******************************************************************************************************************
 * SelectionSortPA encapsulates the entry point of this program, which gets an array size from a command-line argument,
 * creates an array of random integers based on the input size between zero and the maximum integer, displays the array
 * and indicates whether or not the array is sorted, executes and times a selection sort, displays the array after each
 * iteration of the sort, and displays the array and indicates whether or not the array is sorted.
 *
 * @author Tom Lever
 * @version 1.0
 * @since 06/06/21
 ******************************************************************************************************************* */

public class SelectionSortPA 
{
	
	private static int[] theArrayOfIntegers;
	
	
	/** ---------------------------------------------------------------------------------------------------------------
	 * main is the entry point of this program, which gets an array size from a command-line argument, creates an array
	 * of random integers based on the input size between zero and the maximum integer, displays the array and
	 * indicates whether or not the array is sorted, executes and times a selection sort, displays the array after each
	 * iteration of the sort, and displays the array and indicates whether or not the array is sorted.
	 * 
	 * @param args
	 * @throws AnInvalidArraySizeException
	 -------------------------------------------------------------------------------------------------------------- */
	
    public static void main( String[] args ) throws AnInvalidArraySizeException
    {

    	int theArraySize = TheInputAndOutputManager.providesTheArraySizeAsAnIntegerBasedOn(args[0]);
    	
    	theArrayOfIntegers = new int[theArraySize];
    	
    	RandomDataGenerator theRandomDataGenerator = new RandomDataGenerator();
    	for (int i = 0; i < theArraySize; i++)
    	{
    		//theArrayOfIntegers[i] = theRandomDataGenerator.nextInt(0, Integer.MAX_VALUE - 1);
    		theArrayOfIntegers[i] = theRandomDataGenerator.nextInt(0, 10);
    	}
    	
		System.out.println("The array to sort: " + Arrays.toString(theArrayOfIntegers));
		System.out.println("The number of elements in the array: " + theArrayOfIntegers.length);
		TheInputAndOutputManager.printsWhetherOrNotIsSorted(theArrayOfIntegers);
		
		System.out.println("\nExecuting a selection sort.");
		long theStartTime = System.nanoTime();
		selectionSort(theArrayOfIntegers);
		long theEndTime = System.nanoTime();
		System.out.println("The merge sort (with printing) took " + (theEndTime - theStartTime) + " nanoseconds.");	
	
		System.out.println("\nThe array after sorting: " + Arrays.toString(theArrayOfIntegers));
		TheInputAndOutputManager.printsWhetherOrNotIsSorted(theArrayOfIntegers);
		
    }
    
    
    
    /** -------------------------------------------------------------------------------------------------------------
     * selectionSort performs a selection sort of an array of integers and displays the array after each iteration of
     * the sort.
     * 
     * @param arr
     ------------------------------------------------------------------------------------------------------------- */
    
    public static void selectionSort(int[] arr)
    {
    	// Refer to the program for MergeSort in Module 3B Guided Assignment: Problem 2.
    	
    	int theIndexOfTheLastIntegerInTheArray = arr.length - 1;
    	for (int i = 0; i < theIndexOfTheLastIntegerInTheArray; i++)
    	{
    		swapTheIntegersInTheArrayAt(
    			i, getTheIndexOfTheMinimumIntegerInTheSubArrayInclusivelyBetween(i, theIndexOfTheLastIntegerInTheArray)
    		);
    		
    		System.out.println(
    			"\tThe array of integers after iteration " + i + " of Selection Sort: " +
    			Arrays.toString(theArrayOfIntegers)
    		);
    	}
    	
    }
    
    
    /** ---------------------------------------------------------------------------------------------------------
     * getTheIndexOfTheMinimumIntegerInTheSubArrayInclusivelyBetween gets the index of the minimum integer in the
     * subarray of the main array of integers between a first index and a second index.
     * 
     * @param theIndexOfTheFirstIntegerInTheSubArray
     * @param theIndexOfTheLastIntegerInTheSubArray
     * @return
     --------------------------------------------------------------------------------------------------------- */
    
    private static int getTheIndexOfTheMinimumIntegerInTheSubArrayInclusivelyBetween(
    	int theIndexOfTheFirstIntegerInTheSubArray, int theIndexOfTheLastIntegerInTheSubArray
    )
    {
    	int theIndexOfTheMinimumIntegerInTheSubArray = theIndexOfTheFirstIntegerInTheSubArray;
    	
    	for (int j = theIndexOfTheFirstIntegerInTheSubArray + 1; j <= theIndexOfTheLastIntegerInTheSubArray; j++)
    	{
    		if (theArrayOfIntegers[j] < theArrayOfIntegers[theIndexOfTheMinimumIntegerInTheSubArray])
    		{
    			theIndexOfTheMinimumIntegerInTheSubArray = j;
    		}
    	}
    	
    	return theIndexOfTheMinimumIntegerInTheSubArray;
    	
    }
    
    
    /** ----------------------------------------------------------------------------------------------------
     * swapTheIntegersInTheArrayAt swaps the integers in the main array at a first index and a second index.
     * 
     * @param theFirstIndex
     * @param theSecondIndex
     --------------------------------------------------------------------------------------------------- */
    
    private static void swapTheIntegersInTheArrayAt(int theFirstIndex, int theSecondIndex)
    {
    	int thePlaceholder = theArrayOfIntegers[theFirstIndex];
    	
    	theArrayOfIntegers[theFirstIndex] = theArrayOfIntegers[theSecondIndex];
    	
    	theArrayOfIntegers[theSecondIndex] = thePlaceholder;
    }
    
    
    /** ----------------------------------------------------
     * isSorted indicates whether or not an array is sorted.
     * 
     * @param arr
     * @return
     --------------------------------------------------- */
    
    public static boolean isSorted(int[] theArray)
    {
        for(int i = 1; i < theArray.length; i++)
        {
            if(theArray[i] < theArray[i-1]) {
            	return false;
            }
     	}
     	
        return true;
    }
    
}
