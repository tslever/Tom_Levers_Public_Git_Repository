//---------------------------------------------------------------------------
// UseSortedABPriQ.java            by Dale/Joyce/Weems              Chapter 9
//
// Example use of the SortedABPriQ
//---------------------------------------------------------------------------
package ch09.apps;
import ch09.priorityQueues.*;

public class UseSortedABPriQ
{
  public static void main(String[] args)
  { 
    PriQueueInterface<String> pq = new SortedABPriQ<String>();
    
    pq.enqueue("C");   pq.enqueue("O");   pq.enqueue("M");
    pq.enqueue("P");   pq.enqueue("U");   pq.enqueue("T");
    pq.enqueue("E");   pq.enqueue("R");
   
    System.out.println(pq);

    System.out.println(pq.dequeue());
    System.out.println(pq.dequeue());

    System.out.println(pq);
  }
}