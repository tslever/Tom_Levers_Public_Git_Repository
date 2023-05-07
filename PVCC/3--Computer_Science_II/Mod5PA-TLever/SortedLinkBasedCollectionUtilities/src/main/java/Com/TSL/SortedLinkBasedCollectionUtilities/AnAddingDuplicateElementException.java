package Com.TSL.SortedLinkBasedCollectionUtilities;


/**
 * AnAddingDuplicateElementException represents the structure for an exception that occurs when adding a duplicate
 * element to a sorted singly linked list based collection is requested. 
 * 
 * @author Tom Lever
 * @since 05/27/21
 */

class AnAddingDuplicateElementException extends RuntimeException {


  /**
   * AnAddingDuplicateElementException() is a conventional zero-parameter constructor for
   * AnAddingDuplicateElementException, which calls RuntimeException's zero-parameter constructor.
   */

  public AnAddingDuplicateElementException() {
    super();
  }


  /**
   * AnAddingDuplicateElementException(String message) is a one-parameter constructor for
   * AnAddingDuplicateElementException, which passes an error message to RuntimeException's one-parameter constructor
   * with a message parameter.
   * 
   * @param message
   */
  
  protected AnAddingDuplicateElementException(String message) {
    super(message);
  }
  
}