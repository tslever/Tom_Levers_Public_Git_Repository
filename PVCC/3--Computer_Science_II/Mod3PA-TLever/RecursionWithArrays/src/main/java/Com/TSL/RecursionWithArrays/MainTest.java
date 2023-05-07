package Com.TSL.RecursionWithArrays;


import java.util.Arrays;
import org.junit.jupiter.api.Test;


/** ***********************************************************************************************************
 * MainTest encapsulates a JUnit test of core functionality of the method main of class RecursiveMethodsArrays.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *********************************************************************************************************** */

public class MainTest {

	
	/** --------------------------------------------------------------------------------------------
	 * testMainWithValidArgumentsAndAnEmptyArray tests main with valid arguments and an empty array.
	 -------------------------------------------------------------------------------------------- */
	
	@Test
	public void testMainWithValidArgumentsAndAnEmptyArray ()
	{
		
		System.out.println("Running testMainWithValidArgumentsAndAnEmptyArray.");
		
		int[] arr = {};
		
		try {
	        if (arr.length == 0)
	        {
	    	    throw new ANoMinimumExistsException (
	    	    	"Exception: No minimum exists in array arr = " + Arrays.toString(arr) + ".");
	        }
	        
	        System.out.println("A minimum exists in array " + Arrays.toString(arr) + ".");
		}
		
		catch (ANoMinimumExistsException theNoMinimumExistsException)
		{
			System.out.println(theNoMinimumExistsException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	/** ----------------------------------------------------------------------------------------------------------------
	 * testMainWithANumberOfInputArgumentsOtherThanTwo tests the method main with a number of input arguments other than
	 * 2.
	 ---------------------------------------------------------------------------------------------------------------- */
	
	@Test
	public void testMainWithANumberOfInputArgumentsOtherThanTwo ()
	{
		
		System.out.println("Running testMainWithANumberOfInputArgumentsOtherThanTwo.");
		
        String[] args = "I 8 s".split(" ");
        
        try {
	        if (args.length != 2)
	        {
	        	throw new IllegalArgumentException("Exception: The number of input arguments is not equal to 2.");
	        }
	        
	        System.out.println("The number of input arguments is equal to 2.");
        }
        
        catch (IllegalArgumentException theIllegalArgumentException)
        {
        	System.out.println(theIllegalArgumentException.getMessage());
        }
        
        System.out.println();
		
	}
	
	
	/** --------------------------------------------------------------
	 * testMain tests the method main an invalid number of repetitions.
	 -------------------------------------------------------------- */
	
	@Test
	public void testMainWithAnInvalidNumberOfRepetitions ()
	{
		
		System.out.println("Running testMainWithAnInvalidNumberOfRepetitions.");
		
        String[] args = "I s s".split(" ");
        
        try {
        	System.out.println(RecursiveMethodsArrays.repeat(args[0], Integer.parseInt(args[1])));
        	
        	System.out.println("Completed repeat.");
        }
        
        catch (IllegalArgumentException theIllegalArgumentException)
        {
        	System.out.println(theIllegalArgumentException.getMessage());
        }
        
        catch (NumberFormatException theNumberFormatException)
        {
        	System.out.println(theNumberFormatException.getMessage());
        }
		
        System.out.println();
        
	}
	
}
