package com.tsl.playing_with_numbers;


/**
 * StackOverflowException represents the structure for an exception that occurs when a push onto a stack is requested,
 * but the stack is full.
 * @author Tom Lever
 * @version 1.0
 * @since 05/29/21
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