package com.tsl.playing_with_numbers;


/**
 * App encapsulates the entry point to this program that creates an full (i.e., at-capacity) ArrayList-based bounded
 * stack of random integers and outputs statistics relating to the stack.
 * @author Tom Lever
 * @version 1.0
 * @since 05/29/21
 *
 */
public class App {
	
	
	/**
	 * main is the entry point to this program that creates a full ArrayList-based bounded stack of random integers and
	 * outputs statistics relating to the stack. An integer-overflow exception occurs if a range for random integers 
	 * specified within the program is too wide. A stack-overflow exception occurs if the program fills the created
	 * stack beyond the user-specified capacity. A no minimum integer exists exception occurs if the created stack is
	 * empty; throwing of this exception precludes "no maximum integer exists" and "no average exists" exceptions
	 * from occurring.
	 * 
	 * @param args
	 * @throws AnIntegerOverflowException
	 * @throws AStackOverflowException
	 * @throws ANoMinimumIntegerExistsException
	 * @throws ANoMaximumIntegerExistsException
	 * @throws ANoAverageExistsException
	 */
    public static void main( String[] args ) throws
    	AnIntegerOverflowException,
    	AStackOverflowException,
    	ANoMinimumIntegerExistsException,
    	ANoMaximumIntegerExistsException,
    	ANoAverageExistsException
    {
    	
        AFullArrayListBasedBoundedStackOfRandomIntegers theFullArrayListBasedBoundedStackOfRandomIntegers =
        	new AFullArrayListBasedBoundedStackOfRandomIntegers(AnInputOutputManager.askAboutAndReadAUsersAge());
        
        System.out.println("The integers in the stack are:\n" + theFullArrayListBasedBoundedStackOfRandomIntegers);
        
        System.out.println(
        	"The minimum integer in the stack is: " +
        	theFullArrayListBasedBoundedStackOfRandomIntegers.getTheMinimumInteger()
        );
        
        System.out.println(
        	"The maximum integer in the stack is: " +
        	theFullArrayListBasedBoundedStackOfRandomIntegers.getTheMaximumInteger()
        );
        
        System.out.println(
        	"The average of the integers in the stack is: " +
        	theFullArrayListBasedBoundedStackOfRandomIntegers.getTheAverageOfTheIntegers()
        );
        
        System.out.println(
        	"The number of odd integers in the stack is: " +
        	theFullArrayListBasedBoundedStackOfRandomIntegers.getTheNumberOfOddIntegers()
        );
        
    }
}
