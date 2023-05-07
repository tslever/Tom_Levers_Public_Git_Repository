package Com.TSL.MergeSortUtilities;


import java.util.Arrays;
import org.apache.commons.math3.random.RandomDataGenerator;


/** ****************************************************************************************************************
 * MergeSort encapsulates the entry point of this program, which creates an array of integers based on command-line
 * arguments; displays the array, its length, and whether or not it is sorted; performs a merge sort; and, after the
 * sort, displays the array and whether or not it is sorted.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/05/21
 **************************************************************************************************************** */

class MergeSort
{

	/** -----------------------------------------------------------------------------------------------------------
	 * main is the entry point of this program, which creates an array of integers based on command-line arguments;
	 * displays the array, its length, and whether or not it is sorted; performs a merge sort; and, after the sort,
	 * displays the array and whether or not it is sorted.
	 * 
	 * @param args
	 * @throws AnInvalidArraySizeException
	 * @throws AnInvalidLimitsException
	 * @throws AnIntegerOverflowException
	 ---------------------------------------------------------------------------------------------------------- */
	
	public static void main (String[] args)
		throws AnInvalidArraySizeException, AnInvalidLimitsException
	{
		int theArraySize = TheInputAndOutputManager.providesTheArraySizeAsAnIntegerBasedOn(args[0]);
		int theLowerLimitForAnInteger = Integer.parseInt(args[1]);
		int theUpperLimitForAnInteger = Integer.parseInt(args[2]);
		
		if (theLowerLimitForAnInteger > theUpperLimitForAnInteger)
		{
			throw new AnInvalidLimitsException(
				"Exception: The desired lower limit for an integer is greater than the desired upper limit.");
		}
		
		
		int[] theArrayToSort = new int[theArraySize];
		
		RandomDataGenerator theRandomDataGenerator = new RandomDataGenerator();
		for (int i = 0; i < theArraySize; i++)
		{
			theArrayToSort[i] = theRandomDataGenerator.nextInt(theLowerLimitForAnInteger, theUpperLimitForAnInteger);
		}
		
		System.out.println("The array to sort: " + Arrays.toString(theArrayToSort));
		System.out.println("The number of elements in the array: " + theArrayToSort.length);
		TheInputAndOutputManager.printsWhetherOrNotIsSorted(theArrayToSort);
		System.out.println("Executing a merge sort.");
		long theStartTime = System.nanoTime();
		mergeSort(theArrayToSort);
		long theEndTime = System.nanoTime();
		System.out.println("The array after sorting: " + Arrays.toString(theArrayToSort));
		TheInputAndOutputManager.printsWhetherOrNotIsSorted(theArrayToSort);
		System.out.println("The merge sort took " + (theEndTime - theStartTime) + " nanoseconds.");
		
	}
	
	
	/** ------------------------------------------------------
	 * mergeSort performs a merge sort of an array of integers.
	 * 
	 * @param arr
	 ------------------------------------------------------- */
	
	public static void mergeSort(int[] arr)
	{
		
		mergeSortRec(arr, 0, arr.length-1);
		
	}
	
	
	/** -------------------------------------------------------------------------------------------------------
	 * mergeSortRec recursively performs a merge sort of a subarray of integers between two integers inclusive.
	 * 
	 * @param arr
	 * @param first
	 * @param last
	 ------------------------------------------------------------------------------------------------------- */
	
	private static void mergeSortRec(int[] arr, int first, int last)
	{
		
		if(first < last)
		{
			int mid = (first + last) / 2;
			
			mergeSortRec(arr, first, mid);
			
			mergeSortRec(arr, mid + 1, last);
			
			merge(arr, first, mid, mid + 1, last);
		}
		
	}
	
	
	/** -------------------------------------------------------------------------------------------------------------
	 * merge merges a sorted subarray of integers between two integers inclusive with another sorted subarray between
	 * another two integers inclusive.
	 * 
	 * @param arr
	 * @param leftFirst
	 * @param leftLast
	 * @param rightFirst
	 * @param rightLast
	 ------------------------------------------------------------------------------------------------------------ */
	
	private static void merge(int[] arr, int leftFirst, int leftLast, int rightFirst, int rightLast)
	{
		
		int[] aux = new int[arr.length];
	 	//extra space, this is downside of this algorithm
		
		int index = leftFirst;
		
		int saveFirst = leftFirst;
		
		while(leftFirst <= leftLast && rightFirst <= rightLast)
		{
			
			if(arr[leftFirst] <= arr[rightFirst])
			{
				aux[index++] = arr[leftFirst++];
			}
			else
			{
				aux[index++] = arr[rightFirst++];
			}
		}
		
		while(leftFirst <= leftLast)
		{
			aux[index++] = arr[leftFirst++];
		}
		
		while (rightFirst <= rightLast)
		{
			aux[index++]=arr[rightFirst++];
		}
		
		for(index=saveFirst; index<=rightLast; index++)
		{
			arr[index]=aux[index];
		}
		
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