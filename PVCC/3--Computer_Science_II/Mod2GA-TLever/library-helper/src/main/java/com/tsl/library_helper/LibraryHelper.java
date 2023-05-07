package com.tsl.library_helper;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-05
*
* Student name: Tom Lever
* Completion date: 05/27/21
*
*	LibraryHelper.txt: save it as LibraryHelper.java
* 	Implements StackInterface using an array to hold the stack elements.
*
*	Two constructors are provided: one that creates an array of a
*	default size and one that allows the calling program to
*	specify the size.
*
* 	Student tasks: complete tasks specified in the file
*/

public class LibraryHelper<T> implements StackInterface<T> {

	protected final int DEFCAP = 100; // default capacity
	protected T[] elements;           // holds stack elements
	protected int topIndex = -1;      // index of top element in stack

	
	public LibraryHelper() {
	/**
	 * LibraryHelper() is a zero-parameter constructor for LibraryHelper that sets this stack's array of elements to a
	 * new array of objects of type T, with a number of elements equal to this stack's default capacity.
	 */
		
		this.elements = (T[]) new Object[this.DEFCAP];
		
	}

	
	public LibraryHelper(int maxSize) {
	/**
	 * LibraryHelper(int maxSize) is a one-parameter constructor for LibraryHelper that sets this stack's array of
	 * elements to a new array of objects of type T, with a number of elements equal to argument maxSize.
	 * @param maxSize
	 */
		
		//*** Task #1: implement this constructor
		this.elements = (T[]) new Object[maxSize];
		
	}

	
	public void push(T element) {
	// Throws StackOverflowException if this stack is full,
	// otherwise places element at the top of this stack.

		//*** Task #2: implement this method throwing the right exception if necessary
		if (isFull()) {
			throw new StackOverflowException(
				"Exception: push onto a stack of type LibraryHelper failed as the stack was full.");
		}
		
		this.topIndex++;
		this.elements[this.topIndex] = element;
	}

	
	public void pop() {
	// Throws StackUnderflowException if this stack is empty,
	// otherwise removes top element from this stack.

		//*** Task #3: implement this method throwing the right exception if necessary
		if (isEmpty()) {
			throw new StackUnderflowException(
				"Exception: pop from a stack of type LibraryHelper failed as the stack was empty.");
		}
		
		this.elements[this.topIndex] = null;
		this.topIndex--;
	}

	
	public T top() {
	// Throws StackUnderflowException if this stack is empty,
	// otherwise returns top element of this stack.

		//*** Task #4: implement this method throwing the right exception if necessary
		if (isEmpty()) {
			throw new StackUnderflowException(
				"Exception: top from a stack of type LibraryHelper failed as the stack was empty.");
		}
		
		return this.elements[this.topIndex];
	}

	
	public boolean isEmpty() {
	// Returns true if this stack is empty, otherwise returns false.

		//*** Task #5: implement this method
		return (this.topIndex == -1);
	}

	
	public boolean isFull() {
	// Returns true if this stack is full, otherwise returns false.

		//*** Task #6: implement this method
		return (this.topIndex == (this.elements.length - 1));
	}
	
}