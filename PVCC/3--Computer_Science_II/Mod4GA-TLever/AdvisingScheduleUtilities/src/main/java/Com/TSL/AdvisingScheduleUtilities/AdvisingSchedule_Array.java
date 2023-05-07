package Com.TSL.AdvisingScheduleUtilities;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-03
*
* Student name:  Tom Lever
* Completion date: 06/09/21
*
* AdvisingSchedule_Array.java
*
* Implements QueueInterface with an array to hold the queue elements.
*
* Two constructors are provided: one that creates a queue of a default
* capacity and one that allows the calling program to specify the capacity.
*/

public class AdvisingSchedule_Array<T> implements QueueInterface<T>
{

  protected final int DEFCAP = 10; // default capacity
  protected T[] elements;           // array that holds queue elements
  protected int numElements = 0;    // number of elements in this queue
  protected int front = 0;          // index of front of queue
  protected int rear;               // index of rear of queue

  
  /**
   * AdvisingSchedule_Array() is a zero-parameter constructor for AdvisingSchedule_Array, which defines this advising
   * schedule's array of elements as a new array of <default capacity> elements of type T, and sets this advising
   * schedule's index for the rear-most element in the array as the default capacity of this advising schedule minus 1.
   */
  
  public AdvisingSchedule_Array()
  {
    this.elements = (T[]) new Object[DEFCAP];
    this.rear = DEFCAP - 1;
  }

  
  /**
   * AdvisingSchedule_Array(int maxSize) is a one-parameter constructor for AdvisingSchedule_Array, which defines this
   * advising schedule's array of elements as a new array of <argument> elements of type T, and sets this advising
   * schedule's index for the rear-most element in the array as the argument minus 1.
   * @param maxSize
   */
  
  public AdvisingSchedule_Array(int maxSize)
  {
	  this.elements = (T[]) new Object[maxSize];
	  this.rear = maxSize - 1;
  }
  
  
  public void enqueue(T element)
  // Throws QueueOverflowException if this queue is full;
  // otherwise, adds element to the rear of this queue.
  {
	  if (isFull()) {
		  throw new QueueOverflowException("Queue Overflow Exception: enqueue was requested for a full queue.");
	  }
	  
	  this.rear = (this.rear + 1) % this.elements.length;
	  this.elements[this.rear] = element;
	  this.numElements++;
  }

  public T dequeue()
  // Throws QueueUnderflowException if this queue is empty;
  // otherwise, removes front element from this queue and returns it.
  {
	  if (isEmpty()) {
		  throw new QueueUnderflowException("Queue Underflow Exception: dequeue was requested for an empty queue.");
	  }
	  
	  T theStorage = this.elements[this.front];
	  this.elements[this.front] = null;
	  this.front = (this.front + 1) % this.elements.length;
	  this.numElements--;
	  
	  return theStorage;
  }

  public boolean isEmpty()
  // Returns true if this queue is empty; otherwise, returns false.
  {
	  return (this.numElements == 0);
  }

  public boolean isFull()
  // Returns true if this queue is full; otherwise, returns false.
  {
	  return (this.numElements == this.elements.length);
  }

  public int size()
  // Returns the number of elements in this queue.
  {
	  return this.numElements;
  }

}