package Com.TSL.RecursiveMultiplicationUtilities;


/** *******************************************************************************************************************
 * RecursiveMultiplier encapsulates the entry point of this program, which displays an elementary multiplication table,
 * using a recursive multiplication method that does not use Java's * operator.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/03/21
 ******************************************************************************************************************* */

public class RecursiveMultiplier
{
	
	/** ------------------------------------------------------------------------------------------------------------
	 * main is the entry point of this program, which displays an elementary multiplication table, using a recursive
	 * multiplication method that does not use Java's '*' operator. main throws ANotSufficientlyImplementedException
	 * if multiplication with a non-positive factor is requested.
	 * 
	 * @param args
	 * @throws ANotSufficientlyImplementedException
	 ------------------------------------------------------------------------------------------------------------ */
	
    public static void main (String[] args) throws ANotSufficientlyImplementedException
    {
    	System.out.println(factorial(5));
    	
    	// final was removed to avoid "Dead code" warning regarding
    	// if ((THE_LOWEST_FACTOR < 1) || (THE_HIGHEST_FACTOR < 1))
    	int THE_LOWEST_FACTOR = 1;
    	int THE_HIGHEST_FACTOR = 9;
    	
    	if ((THE_LOWEST_FACTOR < 1) || (THE_HIGHEST_FACTOR < 1))
    	{
    		throw new ANotSufficientlyImplementedException("Exception: Factors must be positive.");
    	}
    	
    	int product;
    	
        for (int i = THE_LOWEST_FACTOR; i <= THE_HIGHEST_FACTOR; i++)
        {
        	for (int j = THE_LOWEST_FACTOR; j < THE_HIGHEST_FACTOR; j++)
        	{
        		product = (i < j) ?
        			ARecursiveMultiplicationMachine.multipliesRecursively (j, i) :
        			ARecursiveMultiplicationMachine.multipliesRecursively (i, j);
        		
        		System.out.print (product + " ");
        	}
        	
    		product = (i < THE_HIGHEST_FACTOR) ?
        		ARecursiveMultiplicationMachine.multipliesRecursively (THE_HIGHEST_FACTOR, i) :
        		ARecursiveMultiplicationMachine.multipliesRecursively (i, THE_HIGHEST_FACTOR);
    		
        	System.out.println (ARecursiveMultiplicationMachine.multipliesRecursively (i, THE_HIGHEST_FACTOR));
        	
        }
        
    }
    
    
    static int example(int n) {

    	if (n == 0) {
    		return 0;
    	}
    	else {
    		return example(n - 1) + n * n * n;
    	}
	  
	}
    
    

    static int factorial(int n) {
	
	    if (n == 0) {
            return 1;
	    }
        else {
        	System.out.println("About to invoke factorial again.");
            return (n * factorial(n - 1));
        }
	
	}
        
}


/** ******************************************************************************************************************
 * ARecursiveMultiplicationMachine encapsulates a method that multiplies recursively two integers without using Java's
 * '*' operator.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/03/21
 ****************************************************************************************************************** */

class ARecursiveMultiplicationMachine
{	
	
	/** -------------------------------------------------------------------------------------------
	 * multipliesRecursively multiplies recursively two integers without using Java's '*' operator.
	 * 
	 * @param theFirstInteger
	 * @param theSecondInteger
	 * @return
	 * @throws ANotSufficientlyImplementedException
	 ------------------------------------------------------------------------------------------- */
	
    static int multipliesRecursively(int theFirstInteger, int theSecondInteger)
    {	
    	
    	if (theSecondInteger == 1)
    	{
    		return theFirstInteger;
    	}
    	
    	
    	return theFirstInteger + multipliesRecursively(theFirstInteger, theSecondInteger - 1);
    	
    }
	
}