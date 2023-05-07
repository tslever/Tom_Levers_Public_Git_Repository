package Com.TSL.UtilitiesForQueuingAndEvaluatingRandomNumbers;


//----------------------------------------------------------------------------
// QueueInterface.java
//
// Interface for a class that implements a queue of T.
// A queue is a "first in, first out" structure.
//
// Author: Emilia Butu
// Version: 1.0
// Since: 06/09/21
//----------------------------------------------------------------------------

public interface QueueInterface<T>
{
  void enqueue(T element) throws QueueOverflowException;
  // Throws QueueOverflowException if this queue is full;
  // otherwise, adds element to the rear of this queue.

  T dequeue() throws QueueUnderflowException;
  // Throws QueueUnderflowException if this queue is empty;
  // otherwise, removes front element from this queue and returns it.

  boolean isFull();
  // Returns true if this queue is full; otherwise, returns false.

  boolean isEmpty();
  // Returns true if this queue is empty; otherwise, returns false.

  int size();
  // Returns the number of elements in this queue.
}