//----------------------------------------------------------------------------
// DequeInterface.java           by Dale/Joyce/Weems                 Chapter 4
//
// Interface for a class that implements a deque of T.
// A deque is a linear structure allowing insertion/removal at both ends.
//----------------------------------------------------------------------------

package ch04.queues;

public interface DequeInterface<T>
{
  void enqueueFront(T element) throws QueueOverflowException;
  // Throws QueueOverflowException if this queue is full;
  // otherwise, adds element to the front of this queue.

  void enqueueRear(T element) throws QueueOverflowException;
  // Throws QueueOverflowException if this queue is full;
  // otherwise, adds element to the rear of this queue.

  T dequeueFront() throws QueueUnderflowException;
  // Throws QueueUnderflowException if this queue is empty;
  // otherwise, removes front element from this queue and returns it.

  T dequeueRear() throws QueueUnderflowException;
  // Throws QueueUnderflowException if this queue is empty;
  // otherwise, removes rear element from this queue and returns it.

  boolean isFull();
  // Returns true if this queue is full; otherwise, returns false.

  boolean isEmpty();
  // Returns true if this queue is empty; otherwise, returns false.
  
  int size();
  // Returns the number of elements in this queue.
}




