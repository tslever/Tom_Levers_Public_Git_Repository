//----------------------------------------------------------------------------
// PriQueueInterface.java          by Dale/Joyce/Weems               Chapter 9
//
// Interface for a class that implements a priority queue of T.
// The largest element as determined by the indicated   comparison has the
// highest priority.
//
// Null elements are not allowed. Duplicate elements are allowed.
//----------------------------------------------------------------------------

package ch09.priorityQueues;

public interface PriQueueInterface<T>
{
  void enqueue(T element); 
  // Throws PriQOverflowException if this priority queue is full;
  // otherwise, adds element to this priority queue.

  T dequeue();
  // Throws PriQUnderflowException if this priority queue is empty;
  // otherwise, removes element with highest priority from this 
  // priority queue and returns it.

  boolean isEmpty();
  // Returns true if this priority queue is empty; otherwise, returns false.

  boolean isFull();
  // Returns true if this priority queue is full; otherwise, returns false.
 
  int size();
  // Returns the number of elements in this priority queue.
}

