package Com.TSL.UtilitiesForWorkingWithDoubleEndedQueues;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-03
*
* Student name:  Tom Lever
* Completion date: 06/10/21
*
* Implements DequeInterface using a linked list.
*/

public class Deque<T> implements DequeInterface<T>
{
	
	protected LLNode<T> front;     // reference to the front of this queue
	protected LLNode<T> rear;      // reference to the rear of this queue
	protected int size = 0; 	// number of elements in this queue

	
	// constructor
	public Deque()
	{
		this.front = null;
		this.rear = null;
	}


	public boolean isEmpty()
	// Returns true if this queue is empty; otherwise, returns false.
	{
		return (this.front == null);
	}

	public boolean isFull()
	// Returns false - a linked queue is never full.
	{
		return false;
	}

	public int size()
	// Returns the number of elements in this queue.
	{
		return this.size;
	}

	
	/** -------------------------------------
	 * clear trashes all nodes in this deque.
	 ------------------------------------- */
	
	public void clear()
	{
		this.front = null;
		this.rear = null;
	}
	
	
	// insert an element in the beginning
	public void enqueueFront(T element)
	// Adds element to the front of this queue.
	{
		LLNode<T> theLinkedListNodeForTheElement = new LLNode<T>(element);

		if (this.rear == null) {
			this.rear = theLinkedListNodeForTheElement;
		}
		else {
			theLinkedListNodeForTheElement.setLink(this.front);
		}
		
		this.front = theLinkedListNodeForTheElement;
		
		this.size++;
		
	}

	
	// insert an element at the end
	public void enqueueRear(T element)
	// Throws QueueOverflowException if this queue is full;
	// otherwise, adds element to the rear of this queue.
	{
		if (isFull()) {
			throw new QueueOverflowException("Queue Overflow Exception: enqueueRear requested for a full deque.");
		}
		
		LLNode<T> theLinkedListNodeForTheElement = new LLNode<T>(element);
		
		if (this.rear == null) {
			this.front = theLinkedListNodeForTheElement;			
		}
		else {
			this.rear.setLink(theLinkedListNodeForTheElement);
		}
		
		this.rear = theLinkedListNodeForTheElement;
		
		this.size++;
		
	}


	public T dequeueFront() throws QueueUnderflowException
	// Throws QueueUnderflowException if this queue is empty;
	// otherwise, removes front element from this queue and returns it.
	{
		if (isEmpty()) {
			throw new QueueUnderflowException("Queue Underflow Exception: dequeueFront requested for an empty deque.");
		}
		
		T theDataOfTheFrontLinkedListNode = this.front.getData();
		this.front = this.front.getLink();
		if (this.front == null) {
			this.rear = null;
		}
		
		this.size--;
		
		return theDataOfTheFrontLinkedListNode;
	}

	
	public T dequeueRear() throws QueueUnderflowException
	// Throws QueueUnderflowException if this queue is empty;
	// otherwise, removes rear element from this queue and returns it.
	{
		if (isEmpty()) {
			throw new QueueUnderflowException("Queue Underflow Exception: dequeueFront requested for an empty deque.");
		}
		
		T theDataOfTheRearLinkedListNode;
		
		if (this.front.getLink() == null) {
			theDataOfTheRearLinkedListNode = this.front.getData();
			this.front = null;
			this.rear = null;
			return theDataOfTheRearLinkedListNode;
		}
		
		LLNode<T> theCurrentLinkedListNode = this.front;
		while (theCurrentLinkedListNode.getLink() != this.rear) {
			theCurrentLinkedListNode = theCurrentLinkedListNode.getLink();
		}
		
		theDataOfTheRearLinkedListNode = theCurrentLinkedListNode.getLink().getData();
		theCurrentLinkedListNode.setLink(null);
		this.rear = theCurrentLinkedListNode;
		this.size--;
		
		return theDataOfTheRearLinkedListNode;
	}

	// method to check the front element of the queue
	public T peekAtFront()
	{
		//*** Task #10: implement method
		if(isEmpty()) {
			throw new QueueUnderflowException("underflow exception");
		}
		
		return this.front.getData();
	}


	// method to check the end element of the queue
	public T peekAtRear()
	{
		if (isEmpty()) {
			throw new QueueUnderflowException("Queue Underflow Exception: peekAtRear requested for an empty deque.");
		}
		
		return this.rear.getData();
	}

	// method to display the queue, giving a specific message if the queue is empty
	public void display()
	{
		if (isEmpty()) {
			System.out.println("The deque is empty.");
			return;
		}
		
		
		if (this.front.getLink() == null) {
			System.out.println(this.front.getData() + "\n");
			return;
		}
		
		String theRepresentationOfTheDeque = "";
		
		LLNode<T> theCurrentLinkedListNode = this.front;
		while (theCurrentLinkedListNode != null) {
			theRepresentationOfTheDeque += theCurrentLinkedListNode.getData() + "\n";
			theCurrentLinkedListNode = theCurrentLinkedListNode.getLink();
		}
		
		System.out.println(theRepresentationOfTheDeque);

	}

}