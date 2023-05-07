package Com.TSL.MysterySortUtilities;


import java.util.Arrays;
import org.apache.commons.math3.random.RandomDataGenerator;


/** *****************************************************************************************************************
 * MysterySort encapsulates the entry point of this program, which creates an array of random integers, displays the
 * created array, indicates whether or not the created array is already sorted, executes a mystery sort, displays the
 * sorted array, and indicates whether or not the sorted array is actually sorted.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/05/21
 **************************************************************************************************************** */

public class MysterySort
{
	
	private static int theNumberOfComparisons = 0;
	
	
	/** ----------------------------------------------------------------------------------------------------------------
	 * main is the entry point of this program, which creates an array of random integers, displays the created array,
	 * indicates whether or not the created array is already sorted, executes a mystery sort, displays the sorted array,
	 * and indicates whether or not the sorted array is actually sorted.
	 ---------------------------------------------------------------------------------------------------------------- */
	
    public static void main( String[] args ) throws AnInvalidArraySizeException
    {
    	
    	int theArraySize = TheInputAndOutputManager.providesTheArraySizeAsAnIntegerBasedOn(args[0]);
    	
    	
    	int[] theArrayToSort = new int[theArraySize];
    	
    	RandomDataGenerator theRandomDataGenerator = new RandomDataGenerator();
    	for (int i = 0; i < theArraySize; i++)
    	{
    		//theArrayToSort[i] = theRandomDataGenerator.nextInt(0, Integer.MAX_VALUE - 1);
    		theArrayToSort[i] = theRandomDataGenerator.nextInt(0, 10);
    	}
    	System.out.println("The array to sort: " + Arrays.toString(theArrayToSort));
    	TheInputAndOutputManager.printsWhetherOrNotIsSorted(theArrayToSort);
    	
    	
    	System.out.println("Executing mysterySort.");
    	//mysterySort(theArrayToSort);
    	bubbleSmallIntegersLeftThrough(theArrayToSort);
    	System.out.println("The array after sorting: " + Arrays.toString(theArrayToSort));    	
    	TheInputAndOutputManager.printsWhetherOrNotIsSorted(theArrayToSort);
    	System.out.println("mysterySort performed " + theNumberOfComparisons + " comparisons.");
        
    }
    
    
    /** ----------------------------------------------------
     * isSorted indicates whether or not an array is sorted.
     * 
     * @param arr
     * @return
     --------------------------------------------------- */
    
    public static Boolean isSorted(int[] arr)
    {
        for(int i=1; i<arr.length; i++)
        {
            if(arr[i] < arr[i-1]) {
            	return false;
            }
     	}
     	
        return true;
    }
    
    
    /** ------------------------------------------------------------------
     * mysterySort bubbles large elements right through the array to sort.
     * 
     * @param arr
     ------------------------------------------------------------------ */
    
    public static void mysterySort(int[] arr)
    {
        for(int i=0; i<arr.length; i++)
        {
            for(int k=0; k<arr.length-i-1; k++)
            {
            	theNumberOfComparisons++;
            	
                if(arr[k]>arr[k+1])
                {
               	    int hold=arr[k+1];
               	    
               	    arr[k+1]=arr[k];
               	    
               	    arr[k]=hold;
            	}
         	}
            
    		System.out.println(
				"\tThe array of integers after iteration " + i + " of Bubble Sort: " +
				Arrays.toString(arr)
			);
            
      	}
		
    }
    
    
    /** ------------------------------------------------------------------------------------
     * bubbleSmallIntegersLeftThrough bubbles small integers left through the array to sort.
     * 
     * @param theArrayToSort
     ------------------------------------------------------------------------------------ */
    
    public static void bubbleSmallIntegersLeftThrough(int[] theArrayToSort) {
    	
    	for (int i = 0; i < theArrayToSort.length - 1; i++) {
    		
    		for (int j = theArrayToSort.length - 1; j > i; j--) {
    			
    			if (theArrayToSort[j] < theArrayToSort[j - 1]) {
    				
    				int thePlaceholder = theArrayToSort[j];
    				
    				theArrayToSort[j] = theArrayToSort[j - 1];
    				
    				theArrayToSort[j - 1] = thePlaceholder;
    				
    			}
    			
    		}
    		
    		System.out.println(
				"\tThe array of integers after iteration " + i + " of Bubble Sort: " +
				Arrays.toString(theArrayToSort)
			);
    		
    	}
    	
    }
        
}
