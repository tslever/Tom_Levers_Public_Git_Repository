package Com.TSL.SortedDoublyLinkedListBasedCollectionUtilities;


/**
 * ACollectionUnderflowException represents the structure for an exception that occurs when remove for an empty
 * Collection is requested.
 * 
 * @author Tom Lever
 * @since 05/27/21
 */

class ACollectionUnderflowException extends RuntimeException {


  /**
   * ACollectionUnderflowException() is a conventional zero-parameter constructor for CollectionUnderflowException,
   * which calls Exception's zero-parameter constructor.
   */

  public ACollectionUnderflowException() {
    super();
  }


  /**
   * ACollectionUnderflowException(String message) is a one-parameter constructor for ACollectionUnderflowException,
   * which passes an error message to Exception's one-parameter constructor with a message parameter.
   * 
   * @param message
   */
  
  protected ACollectionUnderflowException(String message) {
    super(message);
  }
  
}