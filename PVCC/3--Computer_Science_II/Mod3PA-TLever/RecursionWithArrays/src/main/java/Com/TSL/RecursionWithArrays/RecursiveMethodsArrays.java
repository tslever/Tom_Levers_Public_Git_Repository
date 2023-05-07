package Com.TSL.RecursionWithArrays;


import java.util.Arrays;



/** *******************************************************************************************************************
 * RecursiveMethodsArrays encapsulates the entry point of this program, which finds the minimum integer in a
 * one-dimensional array of integers, finds the minimum integer in a one-dimensional array of one-dimensional arrays of
 * integers, and concatenates a number of instances of an input string.
 * 
 * @author Yingjin Cui, Tom Lever
 * @version 1.0
 * @since 06/05/21
 ******************************************************************************************************************* */

public class RecursiveMethodsArrays
{
    
	/** ---------------------------------------------------------------------------------------------------------------
	 * main is the entry point of this program, which finds the minimum integer in a one-dimensional array of integers,
	 * finds the minimum integer in a one-dimensional array of one-dimensional arrays of integers, and concatenates a
	 * number of instances of an input string.
	 * 
	 * @param args
	 * @throws ANoMinimumExistsException
	 * @throws IllegalArgumentException
	 -------------------------------------------------------------------------------------------------------------- */
	
    public static void main (String[] args) throws ANoMinimumExistsException, IllegalArgumentException
    {
    	
        int[] arr = {2, 4, 3, 89, 0, -9};
        
        if (arr.length == 0)
        {
    	    throw new ANoMinimumExistsException ("Exception: No minimum exists in array arr = " + Arrays.toString(arr));
        }

        System.out.println(smallest(arr));

        int[][] ar = {{1, 2, 3, 4, 1, 0}, {0, -8, -90}};

        if (ar.length == 0)
        {
    	    throw new ANoMinimumExistsException (
    	    	"Exception: No minimum exists in array arr = " + Arrays.toString (arr)
    	    );
        }
        
        System.out.println(smallest(ar));
        
        if (args.length != 2)
        {
        	throw new IllegalArgumentException("Exception: The number of input arguments is not equal to 2.");
        }
        
        try
        {
        	System.out.println(repeat(args[0], Integer.parseInt(args[1])));
        }
        catch (IllegalArgumentException theIllegalArgumentException)
        {
        	System.out.println(theIllegalArgumentException.getMessage());
        }
    	
	  
        System.out.println();
        
        
        arr = new int[] {5, 6, 90, 1, 8, 2, 9};
      
        if (arr.length == 0)
        {
    	    throw new ANoMinimumExistsException ("Exception: No minimum exists in array arr = " + Arrays.toString(arr));
        }
      
        System.out.println (smallest (arr));
        
        ar = new int[][] {{1, 2, 3}, {0, -9, 5, 1}};
        
        if (ar.length == 0)
        {
    	    throw new ANoMinimumExistsException (
    	    	"Exception: No minimum exists in array arr = " + Arrays.toString (arr)
    	    );
        }
        
        System.out.println (smallest (ar));
      
    }
   
   
   /** --------------------------------------------------------------------------------------------------------------
    * smallest provides the minimum integer in a one-dimensional array.
    * If there is 1 integer i in the array, that integer i is provided.
    * If there are 2 integers in the main array, smallest calls itself, finding the minimum integer j in the subarray
    * of length 2 - 1 starting at index 0 in the main array. This minimum integer in the subarray j is compared with
    * the last element of the main array i, and the smaller of the two is provided as the minimum integer in the main
    * array.
    * If there are n integers in the main array, smallest calls itself, finding the minimum integer j in the subarray
    * of length n - 1 starting at index 0 in the main array. This minimum integer in the subarray j is compared with
    * the last element in the main array i, and the smaller of the two is provided as the minimum integer in the main
    * array.
    * 
    * @param arr
    * @return
    ------------------------------------------------------------------------------------------------------------- */
   
    public static int smallest (int[] arr) {
        // *** Student task #1 ***  

        /* 
        Requirements: 
        This method finds and returns the smallest value in an int array. 
        To get credit, it MUST be implemented as a recursive method. N
        No credit if implemented with loop.

         *** Enter your code below *** 
         */
    	
        if (arr.length == 1)
	    {
            return arr [0];
        }
	   
	    return Math.min (
            arr [arr.length - 1], smallest (Arrays.copyOfRange (arr, 0 /*inclusive*/, arr.length - 1 /*exclusive*/))
        );
	   
    }
     
   
   /** --------------------------------------------------------------------------------------------------------------
    * smallest provides the minimum integer in a main "vertical" array of "horizontal" arrays.
    * If there is 1 horizontal array in the main vertical array, then the minimum integer of that horizontal array is
    * provided.
    * If there are 2 horizontal arrays in the main vertical array, smallest calls itself, finding the minimum integer
    * in the vertical array of length 2 - 1 starting at index 0 in the main vertical array. The minimum of the last
    * horizontal array in the main vertical array is also found. The smaller of these two minima is provided as the
    * minimum integer in the main vertical array.
    * If there are n horizontal arrays in the main vertical array, smallest calls itself, finding the minimum integer
    * in the vertical array of length n - 1 starting at index 0 in the main vertical array. The minimum of the last
    * horizontal array in the main vertical array is also found. The smaller of these two minima is provided as the
    * minimum integer in the main vertical array.
    * 
    * @param arr
    * @return
    -------------------------------------------------------------------------------------------------------------- */
   
    public static int smallest (int[][] arr) {
        // *** Student task #2 ***  

        /* 
        Requirements: 
        This method finds and returns the smallest value in a 2D int array. 
        To get credit, it MUST be implemented as a recursive method. N
        No credit if implemented with loop.
        
        *** Enter your code below *** 
        */
        
        if (arr.length == 1)
        {
        	return smallest (arr [0]);
        }
               
        return Math.min (
            smallest (arr [arr.length - 1]),
            smallest (Arrays.copyOfRange (arr, 0 /*inclusive*/, arr.length - 1 /*exclusive*/))
        );
        
    }
   
    
    /** ----------------------------------------------------------------------------------------------------------------
     * repeat concatenates a number of instances of an input string, and provides the aggregate string.
     * If the number of instances of the input string is negative, then repeat throws an illegal argument exception.
     * If the number of instances of the input string is 0, then repeat provides an empty string.
     * If the number of instances of the input string is 1, then repeat provides the input string.
     * If the number of instances of the input string is greater than 1 and even, then repeat, by calling itself,
     * creates a string that concatenates half the instances of the input string. repeat then concatenates two instances
     * of the resulting string. repeat provides the result.
     * If the number of instances of the input string is greater than 1 and odd, then repeat calls itself, passing in
     * the input string and that number of instances minus 1. repeat concatenates the result with an instance of the
     * input string. repeat provides the result.
     * 
     * @param s
     * @param n
     * @return
     ---------------------------------------------------------------------------------------------------------------- */
      
    public static String repeat (String s, int n) throws IllegalArgumentException {
        // *** Student task #3 ***  

        /* 
        Requirements: 
        It accepts a string s and an integer n as parameters and 
        that returns a String consisting of n copies of s. 
        For example:
 
        Call                       Value Returned
        repeat("hello", 3)    	   "hellohellohello"
        repeat("this is fun", 1)   "this is fun"
        repeat("wow", 0)           ""
        repeat("hi ho! ", 5)       "hi ho! hi ho! hi ho! hi ho! hi ho! "

        *** Enter your code below *** 
        */
     
    	
        if (n < 0)
        {
        	throw new IllegalArgumentException (
        		"Exception: The number of repetitions of the input string is negative."
        	);
        }
        
        
        if (n == 0)
        {
        	return "[empty string]";
        }
        
        
        if (n == 1)
        {
        	return s;
        }        
        
        
        if (n % 2 == 0)
        {
        	String theResultOfConcatenationOfHalfTheInstances = repeat (s, n / 2);
        	return theResultOfConcatenationOfHalfTheInstances + theResultOfConcatenationOfHalfTheInstances;
        }
        
        
        return s + repeat (s, n - 1);
    
    }
    
}