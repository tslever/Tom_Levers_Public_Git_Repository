//----------------------------------------------------------------------------
// LinkedListReverse.java         by Dale/Joyce/Weems                Chapter 3
//
// Demonstrates both iterative and recursive reverse print of linked lists.
// Tests the code developed for Section 3.7: Removing Recursion
//----------------------------------------------------------------------------
package ch03.apps;

import support.LLNode;
import ch02.stacks.*;

public class LinkedListReverse
{
  static void recRevPrintList(LLNode<String> listRef)
  // Prints the contents of the listRef linked list to standard output
  // in reverse order
  {
    if (listRef != null)
    {
       recRevPrintList(listRef.getLink());
       System.out.println(listRef.getInfo());
    }
  }

  static void iterRevPrintList(LLNode<String> listRef)
  // Prints the contents of the listRef linked list to standard output
  // in reverse order
  {
    StackInterface<String> stack = new LinkedStack<String>();
    
    while (listRef != null) // put info onto the stack
    {
       stack.push(listRef.getInfo());
       listRef = listRef.getLink();
    }

    // Retrieve references in reverse order and print elements
    while (!stack.isEmpty())
    {
      System.out.println(stack.top());
      stack.pop();
    }
  }
 
  public static void main(String[] args)
  {
    System.out.println("\n\nTesting empty list:");
    LLNode<String> emptyList = null;

    System.out.println("\n  recursive print: "); recRevPrintList(emptyList); 
    System.out.println("\n  iterative print: "); iterRevPrintList(emptyList);
    
    System.out.println("\n\nTesting list with: alpha:");
    LLNode<String> temp1;
    temp1 = new LLNode<String>("alpha");
    
    LLNode<String> singletonList = temp1;
    
    System.out.println("\n  recursive print: "); recRevPrintList(singletonList); 
    System.out.println("\n  iterative print: "); iterRevPrintList(singletonList);
     
    System.out.println("\n\nTesting list with: alpha, beta, comma, delta, emma:");
    LLNode<String> temp2, temp3, temp4, temp5;
    temp1 = new LLNode<String>("alpha");
    temp2 = new LLNode<String>("beta");
    temp3 = new LLNode<String>("comma");
    temp4 = new LLNode<String>("delta");
    temp5 = new LLNode<String>("emma");
    
    LLNode<String> multiList = temp1;
    temp1.setLink(temp2);   
    temp2.setLink(temp3);   
    temp3.setLink(temp4);   
    temp4.setLink(temp5);  
    
    System.out.println("\n  recursive print: "); recRevPrintList(multiList); 
    System.out.println("\n  iterative print: "); iterRevPrintList(multiList);   
  }
}
