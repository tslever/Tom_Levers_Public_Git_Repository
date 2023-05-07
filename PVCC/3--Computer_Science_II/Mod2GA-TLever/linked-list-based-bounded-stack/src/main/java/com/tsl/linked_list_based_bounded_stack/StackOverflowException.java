package com.tsl.linked_list_based_bounded_stack;


/**
 * StackOverflowException represents the structure for an exception that occurs when a push onto a stack is requested,
 * but the stack is full.
 * @author Tom Lever
 * @version 1.0
 * @since 05/28/21
 *
 */

public class StackOverflowException extends RuntimeException {

	
  public StackOverflowException() {
  /**
   * StackOverflowException() is a conventional zero-parameter constructor for StackOverflowException, which calls
   * Exception's zero-parameter constructor.
   */
	  
    super();
    
  }
  

  public StackOverflowException(String message) {
  /**
   * StackOverflowException(String message) is a one-parameter constructor for StackOverflowException, which passes an
   * error message to Exception's one-parameter constructor with a message parameter.
   * @param message
   */
	  
    super(message);
    
  }
  
}