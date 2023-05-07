package com.tsl.grading_exams;


/**
 * ANoExamScoresExistToAverageException represents the structure for an exception that occurs when an arithmetic
 * operation would cause an integer to be greater than the maximum integer or less than the minimum integer.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 */
class ANoExamScoresExistToAverageException extends Exception {

	/**
	 * ANoExamScoresExistToAverageException() is a conventional zero-argument constructor for
	 * ANoExamScoresExistToAverageException, which calls Exception's zero-argument constructor.
	 */
	protected ANoExamScoresExistToAverageException() {
		super();
	}
	
	
	/**
	 * ANoExamScoresExistToAverageException(String message) is a one-argument constructor for
	 * ANoExamScoresExistToAverageException, which passes an error message to Exception's one-argument constructor with
	 * a message argument.
	 * @param message
	 */
	protected ANoExamScoresExistToAverageException(String message) {
		super(message);
	}
	
}
