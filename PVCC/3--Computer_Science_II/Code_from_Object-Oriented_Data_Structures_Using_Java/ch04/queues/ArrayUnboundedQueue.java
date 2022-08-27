//---------------------------------------------------------------------------
// ArrayUnboundedQueue.java      by Dale/Joyce/Weems                Chapter 4
//
// Implements QueueInterface with an array to hold queue elements.
//
// Two constructors are provided; one that creates a queue of a default
// original capacity and one that allows the calling program to specify the 
// original capacity.
//
// If an enqueue is attempted when there is no room available in the array, a
// new array is created, with capacity incremented by the original capacity.
//---------------------------------------------------------------------------

package ch04.queues;

public class ArrayUnboundedQueue<T> implements QueueInterface<T> 
{
  protected final int DEFCAP = 100; // default capacity
  protected T[] elements;           // array that holds queue elements
  protected int origCap;            // original capacity
  protected int numElements = 0;    // number of elements in this queue
  protected int front = 0;          // index of front of queue
  protected int rear;               // index of rear of queue

  public ArrayUnboundedQueue() 
  {
    elements = (T[]) new Object[DEFCAP];
    rear = DEFCAP - 1;
    origCap = DEFCAP;
  }

  public ArrayUnboundedQueue(int origCap) 
  {
    elements = (T[]) new Object[origCap];
    rear = origCap - 1;
    this.origCap = origCap;
  }

  private void enlarge()
  // Increments the capacity of the queue by an amount 
  // equal to the original capacity.
  {
    // create the larger array
    T[] larger = (T[]) new Object[elements.length + origCap];
    
    // copy the contents from the smaller array into the larger array
    int currSmaller = front;
    for (int currLarger = 0; currLarger < numElements; currLarger++)
    {
      larger[currLarger] = elements[currSmaller];
      currSmaller = (currSmaller + 1) % elements.length;
    }
    
    // update instance variables
    elements = larger;
    front = 0;
    rear = numElements - 1;
  }

  public void enqueue(T element)
  // Adds element to the rear of this queue.
  {  
    if (numElements == elements.length) 
      enlarge();

    rear = (rear + 1) % elements.length;
    elements[rear] = element;
    numElements = numElements + 1;
  }

  public T dequeue()
  // Throws QueueUnderflowException if this queue is empty;
  // otherwise, removes front element from this queue and returns it.
  {       
    if (isEmpty())
      throw new QueueUnderflowException("Dequeue attempted on empty queue.");
    else
    {
      T toReturn = elements[front];
      elements[front] = null;
      front = (front + 1) % elements.length;
      numElements = numElements - 1;
      return toReturn;
    }
  }

  public boolean isEmpty()
  // Returns true if this queue is empty; otherwise, returns false.
  {              
    return (numElements == 0);
  }
  
  public boolean isFull()
  // Returns false - an unbounded queue is never full.
  {              
    return false;
  }
  
  public int size()
  // Returns the number of elements in this queue.
  {
    return numElements;
  } 
}