package Com.TSL.UtilitiesForFindingTheMaximumProductOfTwoDigitIntegersThatIsAPalindrome;


/**
 * NumericPalindrome represents the structure of an array-based bounded queue for five or fewer digits of the product
 * of two two-digit integers.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/11/21
 * @param <T>
 */

public class NumericPalindrome<T> {

	
	private T[] elements;
	private int indexOfTheFrontElement;
	private int indexOfTheRearElement;
	private int numberOfElements;
	
	
	/**
	 * NumericPalindrome(int theCapacity) is the one-parameter constructor for NumericPalindrome, which instantiates
	 * this queue's array of elements as an array with the indicated capacity, and sets this queue's index of the rear
	 * element to the capacity minus 1.
	 * 
	 * @param theCapacity
	 */
	
	public NumericPalindrome(int theCapacity) {
		
		this.elements = (T[]) new Object[theCapacity];
		this.indexOfTheFrontElement = 0;
		this.indexOfTheRearElement = theCapacity - 1;
		this.numberOfElements = 0;
		
	}
	
	
	/**
	 * enqueue adds an element to this queue, or throws a queue overflow exception if this queue is full.
	 * 
	 * @param theElement
	 */
	
	public void enqueue(T theElement) {
		
		if (isFull()) {
			throw new AQueueOverflowException("Exception: enqueue for a full queue was requested.");
		}
		
		this.indexOfTheRearElement = (this.indexOfTheRearElement + 1) % this.elements.length;
		this.elements[this.indexOfTheRearElement] = theElement;
		this.numberOfElements++;
		
	}
	
	
	/**
	 * dequeue removes an element from this queue, or throws a queue underflow exception if this queue is empty.
	 * 
	 * @return
	 */
	
	public T dequeue() {
		
		if (isEmpty()) {
			throw new AQueueUnderflowException("Exception: dequeue for an empty queue was requested.");
		}
		
		T theElementToDequeue = this.elements[this.indexOfTheFrontElement];
		this.elements[this.indexOfTheFrontElement] = null;
		this.indexOfTheFrontElement = (this.indexOfTheFrontElement + 1) % this.elements.length;
		this.numberOfElements--;
		
		return theElementToDequeue;
		
	}
	
	
	/**
	 * isEmpty indicates whether or not this queue is empty.
	 * 
	 * @return
	 */
	
	public boolean isEmpty() {
		
		return (this.numberOfElements == 0);
		
	}
	
	
	/**
	 * isFull indicates whether or not this queue is full.
	 * 
	 * @return
	 */
	
	public boolean isFull() {
		
		return (this.numberOfElements == this.elements.length);
		
	}
	
	
	/**
	 * providesItsNumberOfElements provides this queue's number of elements.
	 * 
	 * @return
	 */
	
	public int providesItsNumberOfElements() {
		
		return this.numberOfElements;
		
	}
	
	
	/**
	 * toString provides a string representation of this queue.
	 */
	
	@Override
	public String toString() {
		
		String theRepresentationOfThisQueue = "[";
		
		for (int i = 0; i < this.numberOfElements - 1; i++) {
			theRepresentationOfThisQueue += this.elements[i] + ", ";
		}
		
		theRepresentationOfThisQueue += this.elements[this.numberOfElements - 1] + "]";
		
		return theRepresentationOfThisQueue;
		
	}
	
}
