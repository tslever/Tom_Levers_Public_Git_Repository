package Com.TSL.CollectionAbstractDataTypeUtilities;


/**
 * AnInvalidIndexException represents an exception that occurs when a get method with an invalid index is requested.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/09/21
 */

public class AnInvalidIndexException extends RuntimeException
{
	
	/**
	 * AnInvalidIndexException() is a conventional zero-parameter constructor for AnInvalidIndexException, which calls
	 * Exception's zero-parameter constructor. 
	 */

	public AnInvalidIndexException()
	{
		super();
	}

	
	/**
	 * AnInvalidIndexException(String message) is a one-parameter constructor for AnInvalidIndexException, which passes
	 * argument message to Exception's one-parameter constructor.
	 * @param message
	 */
  
	public AnInvalidIndexException(String message)
	{
		super(message);
	}
	
}
