package Com.TSL.UtilitiesForWorkingWithDoubleEndedQueues;


/**
 * QueueUnderflowException represents an exception that is thrown when an dequeue operation is requested for an empty
 * queue.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/09/21
 */

public class QueueUnderflowException extends RuntimeException
{
	
  /**
   * QueueUnderflowException() is a conventional zero-parameter constructor for QueueUnderflowException, which calls
   * Exception's zero-parameter constructor. 
   */

  public QueueUnderflowException()
  {
    super();
  }

  
  /**
   * QueueUnderflowException(String message) is a one-parameter constructor for QueueUnderflowException, which passes
   * argument message to Exception's one-parameter constructor.
   * @param message
   */
  
  public QueueUnderflowException(String message)
  {
    super(message);
  }
}