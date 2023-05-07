package Com.TSL.InsertionSortUtilities;


import java.util.Arrays;
import java.util.Random;


/**
 * Insertion Sort
 *
 */

public class App 
{
	
    public static void main( String[] args )
    {
    	
    	int[] theArrayToSort = new int[10];
    	
    	Random random = new Random();
    	for (int i = 0; i < theArrayToSort.length; i++) {
    		theArrayToSort[i] = random.nextInt(10); // [0, 10)
    	}
    	
    	System.out.println(Arrays.toString(theArrayToSort));
    	
        
    	for (int i = 0; i < theArrayToSort.length; i++) {
    		
    		for (int j = i; j > 0; j--) {
    			
    			if (theArrayToSort[j] >= theArrayToSort[j - 1]) {
    				break;
    			}
    				
				int thePlaceholder = theArrayToSort[j];
				
				theArrayToSort[j] = theArrayToSort[j - 1];
				
				theArrayToSort[j - 1] = thePlaceholder;
    			
    		}
    		
        	System.out.println("\t" + Arrays.toString(theArrayToSort));
    		
    	}
    	
    	System.out.println(Arrays.toString(theArrayToSort));
    	
    }
}
