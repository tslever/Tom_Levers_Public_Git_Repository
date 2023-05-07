package Com.TSL.UtilitiesForQueuingAndEvaluatingRandomNumbers;


/** ********************************************
* @author EMILIA BUTU
* version 1.0
* since   2020-03
*
* Student name:  Tom Lever
* Completion date: 06/09/21

* Implements QueueInterface using a linked list.
********************************************* */

public class RandomNumbers_Linked<T> implements QueueInterface<T>
{

  protected LLNode<T> front;     // reference to the front of this queue
  protected LLNode<T> rear;      // reference to the rear of this queue
  protected int numElements = 0; // number of elements in this queue

  
  /**
   * RandomNumbers_Linked is the zero-parameter constructor for RandomNumbers_Linked, which sets this queue's
   * front and rear node variables to reference null.
   */
  
  public RandomNumbers_Linked()
  {
    this.front = null;
    this.rear = null;
  }
  
  
  public void enqueue(T element)
  // Adds element to the rear of this queue.
  {
	  LLNode<T> theLinkedListNodeForTheElement = new LLNode<T>(element);
	  
	  if (this.rear == null) {
		  this.front = theLinkedListNodeForTheElement;
	  }
	  else {
		  this.rear.setLink(theLinkedListNodeForTheElement);
	  }
	  
	  this.rear = theLinkedListNodeForTheElement;
	  
	  this.numElements++;
	  
  }

  
  public T dequeue()
  // Throws QueueUnderflowException if this queue is empty;
  // otherwise, removes front element from this queue and returns it.
  {
 		//*** Task #2: implement the method, using LLNode type objects defined with LLNode class
 		//*   if the queue is empty, throw the appropriate exception
	  
	  if (isEmpty()) {
		  throw new QueueUnderflowException("Queue Underflow Exception: dequeue requested for an empty queue.");
	  }
	  
	  T theInformationOfTheFrontLinkedListNode = this.front.getInfo();
	  this.front = this.front.getLink();
	  if (this.front == null) {
		  this.rear = null;
	  }
	  
	  this.numElements--;
	  
	  return theInformationOfTheFrontLinkedListNode;
	  
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
	  return this.numElements;
  }

}