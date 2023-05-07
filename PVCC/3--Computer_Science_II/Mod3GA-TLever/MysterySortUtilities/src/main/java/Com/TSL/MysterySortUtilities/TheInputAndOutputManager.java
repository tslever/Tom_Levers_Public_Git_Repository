package Com.TSL.MysterySortUtilities;


/** ***********************************************************************************************************
 * TheInputManager encapsulates functionality to provide a desired array size based on a command-line argument.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/05/21
 *********************************************************************************************************** */

class TheInputAndOutputManager
{	
	
	/** ---------------------------------------------------------------------------------------------------------
	 * providesTheArraySizeAsAnIntegerBasedOn provides a desired array size as an integer based on a command-line
	 * argument.
	 * 
	 * @param theArraySizeAsAString
	 * @return
	 * @throws AnInvalidArraySizeException
	 --------------------------------------------------------------------------------------------------------- */
	
	static int providesTheArraySizeAsAnIntegerBasedOn(String theArraySizeAsAString) throws AnInvalidArraySizeException
	{
		int theArraySizeAsAnInteger = Integer.parseInt(theArraySizeAsAString);
		
    	if (theArraySizeAsAnInteger < 0)
    	{
    		throw new AnInvalidArraySizeException("Exception: The array size is negative.");
    	}
    	
    	return theArraySizeAsAnInteger;
    	
	}
	
	
    /** ------------------------------------------------------------------
     * printWhetherOrNotIsSorted prints whether or not an array is sorted.
     * 
     * @param theArray
     ------------------------------------------------------------------ */
	
    static void printsWhetherOrNotIsSorted(int[] theArray)
    {
    	System.out.println("The array " + ((MysterySort.isSorted(theArray)) ? "is " : "is not ") + "sorted.");
    }
	
}