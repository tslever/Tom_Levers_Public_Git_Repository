package Com.TSL.MergeSortUtilities;


import java.util.Arrays;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.junit.jupiter.api.Test;


/** ********************************************************************************************************************
 * MainTest encapsulates a JUnit test of the main program that creates arrays of array sizes, lower limits for integers,
 * and upper limits for integers, and performs a merge sort for an array of random integers for each array size and
 * pair of limits.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/05/21
 ******************************************************************************************************************* */

public class MainTest {

	
	private final int THE_LOWER_LIMIT_FOR_AN_INTEGER = -1073741823;
	private final int THE_UPPER_LIMIT_FOR_AN_INTEGER = 1073741823;
	
	
	/** --------------------------------------------------------------------------------------------------------------
	* testMain creates arrays of array sizes, lower limits for integers, and upper limits for integers, and performs a
	* merge sort for an array of random integers for each array size and pair of limits.
	* 
	* @throws AnIntegerOverflowException
	--------------------------------------------------------------------------------------------------------------- */
	
	@Test
	public void testMain()
	{
		
		int[] theArraySizes = {0, 1, 2, 3, 5, 10};
		
		int[] theLowerLimits = new int[5];
		int[] theUpperLimits = new int[theLowerLimits.length];
		
		RandomDataGenerator theRandomDataGenerator = new RandomDataGenerator();
		for (int i = 0; i < theLowerLimits.length; i++)
		{
			theLowerLimits[i] = theRandomDataGenerator.nextInt(Integer.MIN_VALUE, Integer.MAX_VALUE);
			
			theUpperLimits[i] = theRandomDataGenerator.nextInt(
				theLowerLimits[i], Integer.MAX_VALUE
			);
		}
		
		for (int i = 0; i < theArraySizes.length; i++)
		{
			for (int j = 0; j < theLowerLimits.length; j++)
			{
				
				int[] theArrayToSort = new int[theArraySizes[i]];
				
				for (int k = 0; k < theArraySizes[i]; k++)
				{
					theArrayToSort[k] = theRandomDataGenerator.nextInt(
						theLowerLimits[j], theUpperLimits[j]
					);
				}
				
				System.out.println("The array to sort: " + Arrays.toString(theArrayToSort));
				System.out.println("The number of elements in the array: " + theArrayToSort.length);
				TheInputAndOutputManager.printsWhetherOrNotIsSorted(theArrayToSort);
				System.out.println("Executing a merge sort.");
				long theStartTime = System.nanoTime();
				MergeSort.mergeSort(theArrayToSort);
				long theEndTime = System.nanoTime();
				System.out.println("The array after sorting: " + Arrays.toString(theArrayToSort));
				TheInputAndOutputManager.printsWhetherOrNotIsSorted(theArrayToSort);
				System.out.println("The merge sort took " + (theEndTime - theStartTime) + " nanoseconds.");
				
				System.out.println("\n");
				
			}
			
		}
		
	}
	
}
