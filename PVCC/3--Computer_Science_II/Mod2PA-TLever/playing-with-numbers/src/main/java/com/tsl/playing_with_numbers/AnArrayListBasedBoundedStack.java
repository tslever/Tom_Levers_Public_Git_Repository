package com.tsl.playing_with_numbers;


import java.util.ArrayList;


/**
 * AnArrayListBasedBoundedStack represents a structure for an ArrayList-based bounded stack.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 05/28/21
 *
 */
class AnArrayListBasedBoundedStack<T> {

	
	/**
	 * capacity is an attribute of an ArrayList-based bounded stack.
	 */
	protected int capacity;
	
	
	/**
	 * elements is a component of an ArrayList-based bounded stack with elements of type T.
	 */
	protected ArrayList<T> elements;
	
	
	/**
	 * AnArrayListBasedBoundedStack(int theCapacityOfTheStack) is the one-parameter constructor for
	 * AnArrayListBasedBoundedStack that sets this stack's capacity to theCapacityToUse and sets its array of elements
	 * to a new ArrayList of elements of type T.
	 * 
	 * @param theCapacityToUse
	 */
	protected AnArrayListBasedBoundedStack(int theCapacityToUse) {
		
		this.capacity = theCapacityToUse;
		
		this.elements = new ArrayList<T>();
		
	}
	
	
	/**
	 * isEmpty indicates whether or not this stack is empty.
	 * @return
	 */
	protected boolean isEmpty() {
		
		return (elements.size() == 0);
		
	}
	
	
	/**
	 * isFull indicates whether or not this stack is full.
	 * @return
	 */
	private boolean isFull() {
		
		return (elements.size() == this.capacity);
		
	}
	
	
	/**
	 * receives adds an element to this stack's array of elements, or throws a stack-overflow exception if this stack
	 * is full.
	 * @param theElement
	 * @throws AStackOverflowException
	 */
	protected void receives(T theElement) throws AStackOverflowException {
		
		if (isFull()) {
			throw new AStackOverflowException(
				"Exception: An ArrayList-based bounded stack received a request for it to receive a element when the " +
				"stack was full."
			);
		}
		
		this.elements.add(theElement);
		
	}
	
	
	/**
	 * toString represents this stack as a one-dimensional array with format [a, b, c, ..., x, y, z].
	 * @return
	 */
	@Override
	public String toString() {
		
		return this.elements.toString();
		
	}
	
	
}