//----------------------------------------------------------------------
// RepeatStrings2.java        by Dale/Joyce/Weems              Chapter 4
//
// Sample use of the library ArrayDeque. 
// Outputs strings in same order of entry.
//----------------------------------------------------------------------
package ch04.apps;

import java.util.ArrayDeque;
import java.util.Scanner;

public class RepeatStrings2 
{
  public static void main(String[] args)
  {
    Scanner scan = new Scanner(System.in);

    ArrayDeque<String> stringQueue;
    stringQueue = new ArrayDeque<String>(3);
    
    String line;
    
    for (int i = 1; i <= 3; i++)
    {
      System.out.print("Enter a line of text > ");
      line = scan.nextLine();
      stringQueue.addLast(line);
    }
       
    System.out.println("\nOrder is:\n");
    while (!stringQueue.isEmpty())
    {
      line = stringQueue.removeFirst();
      System.out.println(line);
    }
  }
}