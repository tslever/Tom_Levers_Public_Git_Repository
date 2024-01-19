package Com.TSL.UtilitiesForFindingTheMaximumProductOfTwoDigitIntegersThatIsAPalindrome;


/**
 * AnArrayBasedBoundedStack represents the structure for an array-based bounded stack.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/11/21
 */

public class AnArrayBasedBoundedStack<T> {


	protected T[] elements;
	protected int theIndexOfTheTopElement = -1;

	
	/**
	 * AnArrayBasedBoundedStack(int theCapacity) is the one-parameter constructor for AnArrayBasedBoundedStack that sets
	 * this stack's array of elements to a new array of objects of type T, with a number of elements equal to argument
	 * theCapacity.
	 * 
	 * @param theCapacity
	 */
	
	public AnArrayBasedBoundedStack(int theCapacity) {
		
		this.elements = (T[]) new Object[theCapacity];
		
	}

	
	/**
	 * push places an element at the top of this stack, or throws a stack overflow exception if this stack is full.
	 * 
	 * @param element
	 */
	
	public void push(T theElement) {

		if (isFull()) {
			throw new AStackOverflowException("Exception: push for a full stack was requested.");
		}
		
		this.theIndexOfTheTopElement++;
		this.elements[this.theIndexOfTheTopElement] = theElement;
	}

	
	/**
	 * pop removes the top element from this stack, or throws a stack underflow exception if this stack is empty.
	 */
	
	public void pop() {

		if (isEmpty()) {
			throw new AStackUnderflowException("Exception: pop for an empty stack was requested.");
		}
		
		this.elements[this.theIndexOfTheTopElement] = null;
		this.theIndexOfTheTopElement--;
	}

	
	/**
	 * top provides the top element of this stack, or throws a stack underflow exception if this stack is empty.
	 * @return
	 */
	
	public T top() {

		if (isEmpty()) {
			throw new AStackUnderflowException("Exception: top for an empty stack was requested.");
		}
		
		return this.elements[this.theIndexOfTheTopElement];
	}

	
	/**
	 * isEmpty indicates whether or not this stack is empty.
	 * 
	 * @return
	 */
	
	public boolean isEmpty() {

		return (this.theIndexOfTheTopElement == -1);
	}

	
	/**
	 * isFull indicates whether or not this stack is full.
	 * 
	 * @return
	 */
	
	public boolean isFull() {

		return (this.theIndexOfTheTopElement == (this.elements.length - 1));
	}
	
	
	/**
	 * toString provides a string representation of this stack.
	 */
	
	@Override
	public String toString() {
		
		String theRepresentationOfThisStack = "[";
		
		for (int i = 0; i < this.theIndexOfTheTopElement; i++) {
			theRepresentationOfThisStack += this.elements[i] + ", ";
		}
		
		if (this.theIndexOfTheTopElement > -1) {
			theRepresentationOfThisStack += this.elements[this.theIndexOfTheTopElement];
		}
		
		theRepresentationOfThisStack += "]";
		
		return theRepresentationOfThisStack;
		
	}
	
}