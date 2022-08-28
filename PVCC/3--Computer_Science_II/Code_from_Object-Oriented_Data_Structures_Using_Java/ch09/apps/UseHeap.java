//----------------------------------------------------------------------------
// UseHeap.java               by Dale/Joyce/Weems                    Chapter 9
//
// Example of a simple use of the HeapPriQ.
//----------------------------------------------------------------------------
package ch09.apps;

import ch09.priorityQueues.*;

public class UseHeap
{
  public static void main(String[] args)
  { 
    PriQueueInterface<String> h = new HeapPriQ<String>(10);
    h.enqueue("J");
    h.enqueue("A");
    h.enqueue("M");
    h.enqueue("B");
    h.enqueue("L");
    h.enqueue("E");
    
    System.out.println(h);
    
    System.out.println(h.dequeue() + "\n");
    
    System.out.println(h);
  }
}