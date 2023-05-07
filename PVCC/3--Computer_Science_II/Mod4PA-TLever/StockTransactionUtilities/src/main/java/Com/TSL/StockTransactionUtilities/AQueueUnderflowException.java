package Com.TSL.StockTransactionUtilities;


/**
 * QueueUnderflowException represents an exception that is thrown when an dequeue operation for an empty queue is
 * requested.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/09/21
 */

public class AQueueUnderflowException extends RuntimeException
{
	
	/**
	 * QueueUnderflowException() is a conventional zero-parameter constructor for QueueUnderflowException, which calls
	 * Exception's zero-parameter constructor. 
	 */

	public AQueueUnderflowException()
	{
		super();
	}

	
	/**
	 * QueueUnderflowException(String message) is a one-parameter constructor for QueueUnderflowException, which passes
	 * argument message to Exception's one-parameter constructor.
	 * @param message
	 */
  
	public AQueueUnderflowException(String message)
	{
		super(message);
	}
	
}