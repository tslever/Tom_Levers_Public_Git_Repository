package Com.TSL.UtilitiesForFindingTheMaximumProductOfTwoDigitIntegersThatIsAPalindrome;


/**
 * StackUnderflowException represents an exception that is thrown when an enStack operation for a full Stack is requested.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/09/21
 */

public class AStackUnderflowException extends RuntimeException
{
	
	/**
	 * StackUnderflowException() is a conventional zero-parameter constructor for StackUnderflowException, which calls
	 * Exception's zero-parameter constructor. 
	 */

	public AStackUnderflowException()
	{
		super();
	}

	
	/**
	 * StackUnderflowException(String message) is a one-parameter constructor for StackUnderflowException, which passes
	 * argument message to Exception's one-parameter constructor.
	 * @param message
	 */
  
	public AStackUnderflowException(String message)
	{
		super(message);
	}
	
}