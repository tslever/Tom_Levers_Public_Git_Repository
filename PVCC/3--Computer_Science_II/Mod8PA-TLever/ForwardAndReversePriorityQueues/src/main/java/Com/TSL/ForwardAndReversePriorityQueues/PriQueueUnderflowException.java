package Com.TSL.UtilitiesForGeneratingForwardAndReversePriorityQueues;


/**
 * PriQUnderflowException represents the structure of an exception that occurs when a user attempts to dequeue an empty
 * priority queue.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 07/10/21
 */

class PriQUnderflowException extends RuntimeException
{
	
	/**
	 * PriQUnderflowException() is a conventional zero-parameter constructor for PriQUnderflowException, which calls
	 * RuntimeException's zero-parameter constructor. 
	 */
	
  public PriQUnderflowException()
  {
    super();
  }

  
  /**
   * PriQUnderflowException(String message) is a one-parameter constructor for PriQUnderflowException, which passes
   * a provided message to RuntimeException's one-parameter constructor.
   * 
   * @param message
   */
  
  public PriQUnderflowException(String message)
  {
    super(message);
  }
}