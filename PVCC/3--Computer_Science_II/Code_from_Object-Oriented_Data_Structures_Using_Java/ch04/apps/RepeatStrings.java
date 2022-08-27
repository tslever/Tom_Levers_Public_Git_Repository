//----------------------------------------------------------------------
// RepeatStrings.java        by Dale/Joyce/Weems               Chapter 4
//
// Sample use of queue. Outputs strings in same order of entry.
//----------------------------------------------------------------------
package ch04.apps;

import ch04.queues.*;
import java.util.Scanner;

public class RepeatStrings 
{
  public static void main(String[] args)
  {
    Scanner scan = new Scanner(System.in);

    QueueInterface<String> stringQueue;
    stringQueue = new ArrayBoundedQueue<String>(3);
    
    String line;
    
    for (int i = 1; i <= 3; i++)
    {
      System.out.print("Enter a line of text > ");
      line = scan.nextLine();
      stringQueue.enqueue(line);
    }
       
    System.out.println("\nOrder is:\n");
    while (!stringQueue.isEmpty())
    {
      line = stringQueue.dequeue();
      System.out.println(line);
    }
  }
}