//----------------------------------------------------------------------------
// GlassQueueInterface.java         by Dale/Joyce/Weems              Chapter 4
//
// Interface for a class that implements a queue of T and includes operations
// for peeking at the front and rear elements of the queue.
//----------------------------------------------------------------------------

package ch04.queues;

public interface GlassQueueInterface<T> extends QueueInterface<T>
{
  public T peekFront();
  // If the queue is empty, returns null.
  // Otherwise returns the element at the front of this queue.
  
  public T peekRear();
  // If the queue is empty, returns null.
  // Otherwise returns the element at the rear of this queue.
}




