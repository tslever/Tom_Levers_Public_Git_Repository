package Com.TSL.UtilitiesForQueuingAndEvaluatingRandomNumbers;


/**
 * QueueOverflowException represents an exception that is thrown when an enqueue operation is requested for a full queue.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/09/21
 */

public class QueueOverflowException extends RuntimeException
{
	
  /**
   * QueueOverflowException() is a conventional zero-parameter constructor for QueueOverflowException, which calls
   * Exception's zero-parameter constructor. 
   */
	
  public QueueOverflowException()
  {
    super();
  }

  
  /**
   * QueueOverflowException(String message) is a one-parameter constructor for QueueOverflowException, which passes
   * argument message to Exception's one-parameter constructor.
   * @param message
   */
  
  public QueueOverflowException(String message)
  {
    super(message);
  }
}