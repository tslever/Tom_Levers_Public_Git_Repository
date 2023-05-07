package com.tsl.pez_candy;


/**
 * StackUnderflowException represents the structure for an exception that occurs when a pop off a stack is requested,
 * or a look at the top element in the stack is requested, but the stack is empty.
 * @author Tom Lever
 * @since 05/27/21
 *
 */

public class AStackUnderflowException extends Exception {

	
  /**
   * StackUnderflowException() is a conventional zero-parameter constructor for StackUnderflowException, which calls
   * Exception's zero-parameter constructor.
   */
  protected AStackUnderflowException() {
	  
    super();
    
  }

  
  /**
   * StackUnderflowException(String message) is a one-parameter constructor for StackUnderflowException, which passes an
   * error message to Exception's one-parameter constructor with a message parameter.
   * @param message
   */
  protected AStackUnderflowException(String message) {
	  
    super(message);
    
  }
  
}