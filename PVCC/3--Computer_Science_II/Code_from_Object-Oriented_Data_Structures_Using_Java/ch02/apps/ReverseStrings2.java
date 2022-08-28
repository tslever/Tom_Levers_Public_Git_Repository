//----------------------------------------------------------------------
// ReverseStrings2.java       by Dale/Joyce/Weems              Chapter 2
//
// Sample use of the library Stack. 
// Outputs strings in reverse order of entry.
//----------------------------------------------------------------------
package ch02.apps;

import java.util.Stack;
import java.util.Scanner;

public class ReverseStrings2 
{
  public static void main(String[] args)
  {
    Scanner scan = new Scanner(System.in);

    Stack<String> stringStack = new Stack<String>();
    
    String line;
    
    for (int i = 1; i <= 3; i++)
    {
      System.out.print("Enter a line of text > ");
      line = scan.nextLine();
      stringStack.push(line);
    }
    
    System.out.println("\nReverse is:\n");
    while (!stringStack.empty())
    {
      line = stringStack.peek();
      stringStack.pop();
      System.out.println(line);
    }
  }
}