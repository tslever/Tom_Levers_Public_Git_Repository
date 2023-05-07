package com.tsl.pez_candy;


/**
 * StackOverflowException represents the structure for an exception that occurs when a push onto a stack is requested,
 * but the stack is full.
 * @author Tom
 *
 */

public class AStackOverflowException extends Exception {

	
  /**
   * StackOverflowException() is a conventional zero-parameter constructor for StackOverflowException, which calls
   * Exception's zero-parameter constructor.
   */
  protected AStackOverflowException() {
	  
    super();
    
  }
  

  /**
   * StackOverflowException(String message) is a one-parameter constructor for StackOverflowException, which passes an
   * error message to Exception's one-parameter constructor with a message parameter.
   * @param message
   */
  protected AStackOverflowException(String message) {
	  
    super(message);
    
  }
  
}
