//--------------------------------------------------------------------------
// LinkedGlassQueue.java         by Dale/Joyce/Weems               Chapter 4
//
// Extends LinkedQueue with operations to access the front and rear queue 
// elements without removing them.
//---------------------------------------------------------------------------
package ch04.queues;

public class LinkedGlassQueue<T> extends LinkedQueue<T> implements GlassQueueInterface<T>
{
  public LinkedGlassQueue() 
  {
    super();
  }

  public T peekFront()
  // If the queue is empty, returns null.
  // Otherwise returns the element at the front of this queue.
  {
    if (isEmpty())
       return null;
    else 
       return front.getInfo();
  }
  
  public T peekRear()
  // If the queue is empty, returns null.
  // Otherwise returns the element at the rear of this queue.
  {
    if (isEmpty())
       return null;
    else 
       return rear.getInfo();
  }
}