package com.tsl.array_based_bounded_stack;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-03
*
* Student name:  Tom Lever
* Completion date: 05/28/21
*
*	ImprovedArrayBoundedStack.txt: download and save as ImprovedArrayBoundedStack.java
* 	Tasks are marked throughout the file
*
* 	Student tasks: complete tasks specified in the file
*
*/

public class ImprovedArrayBoundedStack<T> implements StackInterface<T>
{
	protected final int DEFCAP = 100; // default capacity
	protected T[] elements;           // holds stack elements
	protected int topIndex = -1;      // index of top element in stack

	
	public ImprovedArrayBoundedStack() {
	/**
	 * ImprovedArrayBoundedStack is a zero-parameter constructor for ImprovedArrayBoundedStack that sets this stack's
	 * array of elements to a new array of objects of type T, with a number of elements equal to this stack's default
	 * capacity.
	 */
		
		elements = (T[]) new Object[DEFCAP];
	}

	public ImprovedArrayBoundedStack(int maxSize) {
	/**
	 * ImprovedArrayBoundedStack is a one-parameter constructor for ImprovedArrayBoundedStack that sets this stack's
	 * array of elements to a new array of objects of type T, with a number of elements equal to argument maxSize.
	 */
		
		elements = (T[]) new Object[maxSize];
	}

	public void push(T element)
	// Throws StackOverflowException if this stack is full,
	// otherwise places element at the top of this stack.
	{
		if (isFull())
			throw new StackOverflowException("Push attempted on a full stack.");
		else
		{
			topIndex++;
			elements[topIndex] = element;
		}
	}

	public void pop()
	// Throws StackUnderflowException if this stack is empty,
	// otherwise removes top element from this stack.
	{
		if (isEmpty())
			throw new StackUnderflowException("Pop attempted on an empty stack.");
		else
		{
			elements[topIndex] = null;
			topIndex--;
		}
	}

	public T top()
	// Throws StackUnderflowException if this stack is empty,
	// otherwise returns top element of this stack.
	{
		T topOfStack = null;
		if (isEmpty())
			throw new StackUnderflowException("Top attempted on an empty stack.");
		else
			topOfStack = elements[topIndex];
		return topOfStack;
	}

	public boolean isEmpty()
	// Returns true if this stack is empty, otherwise returns false.
	{
		return (topIndex == -1);
	}

	public boolean isFull()
	// Returns true if this stack is full, otherwise returns false.
	{
		return (topIndex == (elements.length - 1));
	}

		//*** Task #1: define and implement method toString(): String
		//*	creates and returns a string that correctly represents the current stack.

	@Override
	public String toString() {
	/**
	 * toString arranges strings representing the elements in this stack in an indented column.
	 */
		
		String theStringRepresentingTheStack = "";
		
		for (int i = topIndex; i > 0; i--) {
			theStringRepresentingTheStack += "\t" + this.elements[i].toString() + "\n";
		}
		if (topIndex > -1) {
			theStringRepresentingTheStack += "\t" + this.elements[0].toString();
		}
		
		return theStringRepresentingTheStack;
		
	}

		//*** Task #2: define and implement method size(): int
		//*	returns a count of how many items are currently on the stack.

	public int size() {
	// returns a count of how many elements are on the stack

		return this.topIndex + 1;

	}

		//*** Task #3: define and implement method  popSome(int count): void
		//*	removes the top count elements from the stack

	public void popSome(int count) {
	// if possible, removes top count elements from stack;
	// otherwise throws StackUnderflowException

		if (size() < count) {
			throw new StackUnderflowException(
				"Exception: popSome(count=" + count + ") for an improved array-based bounded stack was requested " +
				"when the number of elements on the stack, size = " + size() + ", was less than count = " + count + "."
			);
		}
		
		while (count > 0) {
			pop();
			count--;
		}

	}

		//*** Task #4: define and implement method  swapStart(): boolean
		//*	 if there are less than two elements on the stack returns false;
		//*	 otherwise it reverses the order of the top two elements on the stack and returns true

	public boolean swapStart() {
	// if possible, reverses order of top 2 elements and returns true;
	// otherwise returns false

		if (size() < 2) {
			return false;
		}
		
		T theElementInStorage = this.elements[topIndex];
		this.elements[topIndex] = this.elements[topIndex - 1];
		this.elements[topIndex - 1] = theElementInStorage;
		return true;

	}
		//*** Task #5: define and implement method poptop( ): T
		//*	 the “classic” pop operation, if the stack is empty it throws StackUnderflowException;
		//*	 otherwise it both removes and returns the top element of the stack.

	public T poptop() {
	// Throws StackUnderflowException if this stack is empty,
	// otherwise removes and returns top element from this stack.

		if (isEmpty()) {
			throw new StackUnderflowException(
				"Exception: poptop for an improved array-based bounded stack was requested when the stack was empty."
			);
		}
		
		T theTopElement = top();
		pop();
		return theTopElement;

	}
	
}