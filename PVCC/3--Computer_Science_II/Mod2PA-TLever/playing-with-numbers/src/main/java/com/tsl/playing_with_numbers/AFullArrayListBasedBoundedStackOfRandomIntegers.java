package com.tsl.playing_with_numbers;


/**
 * AFullArrayListBasedBoundedStackOfRandomIntegers represents the structure for a full (i.e., at-capacity) ArrayList-
 * based bounded stack of random integers.
 * @author Tom Lever
 * @version 1.0
 * @since 05/29/21
 *
 */
class AFullArrayListBasedBoundedStackOfRandomIntegers extends AnArrayListBasedBoundedStack<Integer> {

	
	/**
	 * THE_MINIMUM_INTEGER is an attribute of AFullArrayListBasedBoundedStackOfRandomIntegers.
	 */
	private final int THE_MINIMUM_INTEGER = -2147483648;
	
	
	/**
	 * THE_MAXIMUM_INTEGER is an attribute of AFullArrayListBasedBoundedStackOfRandomIntegers.
	 */
	private final int THE_MAXIMUM_INTEGER = 2147483647;
	
	
	/**
	 * THE_LOWER_LIMIT_FOR_A_RANDOM_NUMBER_GENERATOR is an attribute of
	 * AFullArrayListBasedBoundedStackOfRandomIntegers.
	 */
	private final int THE_LOWER_LIMIT_FOR_A_RANDOM_NUMBER_GENERATOR = 1;
	
	
	/**
	 * THE_UPPER_LIMIT_FOR_A_RANDOM_NUMBER_GENERATOR is an attribute of
	 * AFullArrayListBasedBoundedStackOfRandomIntegers.
	 */
	private final int THE_UPPER_LIMIT_FOR_A_RANDOM_NUMBER_GENERATOR = 1000;
	
	
	/**
	 * AFullArrayListBasedBoundedStackOfRandomIntegers(int theCapacityOfTheStack) is the one-parameter constructor for
	 * AFullArrayListBasedBoundedStackOfRandomIntegers that calls the one-parameter constructor of
	 * AnArrayListBasedBoundedStack and fills this stack with random integers to capacity. An integer-overflow
	 * exception occurs if a range for random integers specified within the program is too wide. A stack-overflow
	 * exception occurs if the program fills the created stack beyond the user-specified capacity.
	 * 
	 * @param theCapacityOfTheStack
	 * @throws AnIntegerOverflowException
	 * @throws AStackOverflowException
	 */
	protected AFullArrayListBasedBoundedStackOfRandomIntegers(int theCapacityToUse)
		throws AnIntegerOverflowException, AStackOverflowException {
		
		super(theCapacityToUse);
		
		this.fillWithRandomIntegersToCapacity();

	}
	
	
	/**
	 * checkTheAdditionOf throws an integer-overflow exception if addition of a first integer and a
	 * second integer would result in a sum greater than the maximum integer or less than the minimum integer.
	 * 
	 * @param theFirstInteger
	 * @param theSecondInteger
	 * @throws AnIntegerOverflowException
	 */
	private void checkTheAdditionOf(int theFirstInteger, int theSecondInteger) throws AnIntegerOverflowException {
		
		if ((theFirstInteger > 0) && (theSecondInteger > 0) && (theFirstInteger > this.THE_MAXIMUM_INTEGER - theSecondInteger) ||
			(theFirstInteger < 0) && (theSecondInteger < 0) && (theFirstInteger < this.THE_MINIMUM_INTEGER - theSecondInteger)) {
			
			throw new AnIntegerOverflowException(
				"Integer-overflow exception: the sum of " + theFirstInteger + " and " + theSecondInteger +
				" is outside the interval [" + this.THE_MINIMUM_INTEGER + ", " + this.THE_MAXIMUM_INTEGER + "]."
			);
			
		}
		
	}	
	
	
	/**
	 * fillWithRandomIntegersToCapacity fills this stack its capacity. An integer-overflow exception occurs if a range
	 * for random integers specified within the program is too wide. A stack-overflow exception occurs if the program
	 * fills the created stack beyond the user-specified capacity.
	 * 
	 * @throws AnIntegerOverflowException
	 * @throws AStackOverflowException
	 */
	private void fillWithRandomIntegersToCapacity() throws AnIntegerOverflowException, AStackOverflowException {
		
		for (int i = 0; i < this.capacity; i++) {
			
			this.receives(
				ARandomNumberGenerator.getARandomIntegerInclusivelyBetween(
					this.THE_LOWER_LIMIT_FOR_A_RANDOM_NUMBER_GENERATOR,
					this.THE_UPPER_LIMIT_FOR_A_RANDOM_NUMBER_GENERATOR)
			);
			
		}
		
	}
	
	
	/**
	 * getTheAverageOfTheIntegers gets the average of the integers in this stack. A no average exists exception occurs
	 * if the stack is empty. An integer-overflow exception occurs if the sum that used in calculating the average
	 * exceeds the maximum integer.
	 * 
	 * @return
	 * @throws ANoAverageExistsException
	 * @throws AnIntegerOverflowException
	 */
	protected double getTheAverageOfTheIntegers() throws ANoAverageExistsException, AnIntegerOverflowException {
		
		if (isEmpty()) {
			throw new ANoAverageExistsException("Exception: No average exists.");
		}
		
		int theSumOfTheIntegersInTheStack = 0;
		
		for (int i = 0; i < this.elements.size(); i++) {
		
			checkTheAdditionOf(theSumOfTheIntegersInTheStack, this.elements.get(i));
			
			theSumOfTheIntegersInTheStack += this.elements.get(i);
			
		}
		
		return (double)theSumOfTheIntegersInTheStack / (double)this.elements.size();
		
	}
	
	
	/**
	 * getTheMaximumInteger gets the maximum integer in this stack. A no maximum integer exists exception occurs if
	 * the stack is empty.
	 * 
	 * @return
	 * @throws ANoMaximumIntegerExistsException
	 */
	protected int getTheMaximumInteger() throws ANoMaximumIntegerExistsException {
		
		if (isEmpty()) {
			throw new ANoMaximumIntegerExistsException("Exception: No maximum integer exists.");
		}
		
		int theMaximumIntegerInTheStack = this.THE_LOWER_LIMIT_FOR_A_RANDOM_NUMBER_GENERATOR;
		
		for (int i = 0; i < this.elements.size(); i++) {
			if (this.elements.get(i) > theMaximumIntegerInTheStack) {
				theMaximumIntegerInTheStack = this.elements.get(i);
			}
		}
		
		return theMaximumIntegerInTheStack;
		
	}
	
	
	/**
	 * getTheMinimumInteger gets the minimum integer in this stack. A no minimum integer exists exception occurs if
	 * the stack is empty.
	 * 
	 * @return
	 * @throws ANoMinimumIntegerExistsException
	 */
	protected int getTheMinimumInteger() throws ANoMinimumIntegerExistsException {
		
		if (isEmpty()) {
			throw new ANoMinimumIntegerExistsException("Exception: No minimum integer exists.");
		}
		
		int theMinimumIntegerInTheStack = this.THE_UPPER_LIMIT_FOR_A_RANDOM_NUMBER_GENERATOR;
		
		for (int i = 0; i < this.elements.size(); i++) {
			if (this.elements.get(i) < theMinimumIntegerInTheStack) {
				theMinimumIntegerInTheStack = this.elements.get(i);
			}
		}
		
		return theMinimumIntegerInTheStack;
		
	}
	
	
	/**
	 * getTheNumberOfOddIntegers gets the number of odd integers in this stack.
	 * 
	 * @return
	 */
	protected int getTheNumberOfOddIntegers() {
		
		int theNumberOfOddIntegers = 0;
		
		for (int i = 0; i < this.elements.size(); i++) {
			
			if (isOdd(this.elements.get(i))) {
				theNumberOfOddIntegers++;
			}
			
		}
		
		return theNumberOfOddIntegers;
		
	}
	
	
	/**
	 * isOdd indicates whether or not its integer argument is odd.
	 * 
	 * @param theInteger
	 * @return
	 */
	private boolean isOdd(int theInteger) {
		
		return (theInteger % 2 != 0);
		
	}
	
}