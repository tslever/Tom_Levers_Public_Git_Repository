package Com.TSL.UtilitiesForFindingTheMaximumProductOfTwoDigitIntegersThatIsAPalindrome;


/**
 * StackOverflowException represents an exception that is thrown when an enStack operation for a full Stack is requested.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/09/21
 */

public class AStackOverflowException extends RuntimeException
{
	
	/**
	 * StackOverflowException() is a conventional zero-parameter constructor for StackOverflowException, which calls
	 * Exception's zero-parameter constructor. 
	 */

	public AStackOverflowException()
	{
		super();
	}

	
	/**
	 * StackOverflowException(String message) is a one-parameter constructor for StackOverflowException, which passes
	 * argument message to Exception's one-parameter constructor.
	 * @param message
	 */
  
	public AStackOverflowException(String message)
	{
		super(message);
	}
	
}