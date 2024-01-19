package Com.TSL.UtilitiesForFindingTheMaximumProductOfTwoDigitIntegersThatIsAPalindrome;


/**
 * QueueOverflowException represents an exception that is thrown when an enqueue operation for a full queue is requested.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/09/21
 */

public class AQueueOverflowException extends RuntimeException
{
	
	/**
	 * QueueOverflowException() is a conventional zero-parameter constructor for QueueOverflowException, which calls
	 * Exception's zero-parameter constructor. 
	 */

	public AQueueOverflowException()
	{
		super();
	}

	
	/**
	 * QueueOverflowException(String message) is a one-parameter constructor for QueueOverflowException, which passes
	 * argument message to Exception's one-parameter constructor.
	 * @param message
	 */
  
	public AQueueOverflowException(String message)
	{
		super(message);
	}
	
}