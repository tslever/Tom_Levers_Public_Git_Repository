package com.tsl.linked_list_based_bounded_stack;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-03
*
* Student name:  Tom Lever
* Completion date: 05/28/21
*/

public class ImprovedLinkedStack<T> implements StackInterface<T>
{
	protected LLNode<T> top;   // reference to the top of this stack

	
	public ImprovedLinkedStack() {
	/**
	 * ImprovedLinkedStack() is a zero-parameter constructor for ImprovedLinkedStack that sets the top linked-list node
	 * of this linked list based stack to null.
	 */
		
		top = null;
	}

	public void push(T element) {
	// Places element at the top of this stack.

		LLNode<T> newNode = new LLNode<T>(element);
		newNode.setsItsReferenceTo(top);
		top = newNode;
		
	}

	public void pop() {
	// Throws StackUnderflowException if this stack is empty,
	// otherwise removes top element from this stack.

		if (isEmpty())
			throw new StackUnderflowException("Pop attempted on an empty stack.");
		else
			top = top.getTheReferenceToTheNextLinkedListNode();
	}

	public T top() {
	// Throws StackUnderflowException if this stack is empty,
	// otherwise returns the information of the top element of this stack.

		if (isEmpty())
			throw new StackUnderflowException("Top attempted on an empty stack.");
		else
			return top.getTheInformation();
	}

	public boolean isEmpty() {
	// Returns true if this stack is empty, otherwise returns false.

		return (top == null);
	}

	public boolean isFull() {
	// Returns false - a linked stack is never full

		return false;
		
	}

		//*** Task #1: define method toString(): String
		//*	creates and returns a string that correctly represents the current stack.


	@Override
	public String toString() {
	/**
	 * toString arranges strings representing the information of the linked-list nodes in this stack in an indented
	 * column.
	 */
		
		String theStringRepresentingTheStack = "";
		
		LLNode<T> theCurrentNode = top;
		theStringRepresentingTheStack += "\t" + top.getTheInformation();
		theCurrentNode = theCurrentNode.getTheReferenceToTheNextLinkedListNode();
		
		while (theCurrentNode != null) {
			theStringRepresentingTheStack += "\n\t" + theCurrentNode.getTheInformation();
			theCurrentNode = theCurrentNode.getTheReferenceToTheNextLinkedListNode();
		}
		
		return theStringRepresentingTheStack;

	}

		//*** Task #2: define method size(): int
		//*	returns a count of how many items are currently on the stack.


	public int size() {
	// returns a count of how many elements are on the stack
		
		int theNumberOfElementsOnTheStack = 0;
		
		LLNode<T> theCurrentNode = top;
		while (theCurrentNode != null) {
			theNumberOfElementsOnTheStack++;
			theCurrentNode = theCurrentNode.getTheReferenceToTheNextLinkedListNode();
		}
		
		return theNumberOfElementsOnTheStack;

	}

		//*** Task #3: define method  popSome(int count): void
		//*	removes the top count elements from the stack

	public void popSome(int count) {
	// if possible, removes top count elements from stack;
	// otherwise throws StackUnderflowException

		if (size() < count) {
			throw new StackUnderflowException(
				"Exception: popSome(count=" + count + ") for an improved linked list based stack was requested when " +
				"the number of elements on the stack, size = " + size() + ", was less than count = " + count + "."
			);
		}
		
		while (count > 0) {
			pop();
			count--;
		}

	}

		//*** Task #4: define method  swapStart(): boolean
		//*	 if there are less than two elements on the stack returns false;
		//*	 otherwise it reverses the order of the top two elements on the stack and returns true

	public boolean swapStart() {
	// if possible, reverses order of top 2 elements and returns true;
	// otherwise returns false
		
		if (size() < 2) {
			return false;
		}
		
		LLNode<T> theNodeInStorage = top.getTheReferenceToTheNextLinkedListNode();
		top.setsItsReferenceTo(top.getTheReferenceToTheNextLinkedListNode().getTheReferenceToTheNextLinkedListNode());
		theNodeInStorage.setsItsReferenceTo(top);
		top = theNodeInStorage;
		return true;
		
	}
		//*** Task #5: define method poptop( ): T
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
		
		T theInformationOfTheTopLinkedListNode = top();
		pop();
		return theInformationOfTheTopLinkedListNode;

	}

}