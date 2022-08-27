//----------------------------------------------------------------------------
// LinkedListRecursive.java        by Dale/Joyce/Weems               Chapter 3
//
// Demonstrates both iterative and recursive traversals of linked lists.
// Demonstrates both iterative and recursive element count of linked lists.
// Demonstrates recursively adding an element to the end of a linked list.
// Tests the code developed for Section 3.4: Linked List Processing
//----------------------------------------------------------------------------
package ch03.apps;

import support.LLNode;

public class LinkedListRecursive
{
  static void recPrintList(LLNode<String> listRef)
  // Prints the contents of the listRef linked list to standard output
  {
    if (listRef != null)
    {
       System.out.println(listRef.getInfo());
       recPrintList(listRef.getLink());
    }
  }

  static void iterPrintList(LLNode<String> listRef)
  // Prints the contents of the listRef linked list to standard output
  {
    while (listRef != null)
    {
       System.out.println(listRef.getInfo());
       listRef = listRef.getLink();
    }
  }

  static int recListSize(LLNode<String> listRef)
  // Returns the size of the listRef linked list
  {
    if (listRef == null)
      return 0;
    else 
      return 1 + recListSize(listRef.getLink());
  }

  static int iterListSize(LLNode<String> listRef)
  // Returns the size of the listRef linked list
  {
    int size = 0;
    while (listRef != null)
    {
      size = size + 1;
      listRef = listRef.getLink();
    }
    return size;
  }

  static LLNode<String> recInsertEnd(String newInfo, LLNode<String> listRef)
  // Adds newInfo to the end of the listRef linked list
  {
    if (listRef != null)
      listRef.setLink(recInsertEnd(newInfo, listRef.getLink()));
    else
      listRef = new LLNode<String>(newInfo);
    return listRef;
  }
 
  public static void main(String[] args)
  {
    System.out.println("\n\nTesting empty list:");
    LLNode<String> emptyList = null;

    System.out.println("\n  recursive print: "); recPrintList(emptyList); 
    System.out.println("\n  iterative print: "); iterPrintList(emptyList);
    System.out.println("\n  recursive size: ");  System.out.println(recListSize(emptyList)); 
    System.out.println("\n  iterative size: "); System.out.println(iterListSize(emptyList)); 
    
    System.out.println("\n  adding \"final\" ... ");
    emptyList = recInsertEnd("final", emptyList);
    System.out.println("\n  recursive print: "); recPrintList(emptyList); 
     
    System.out.println("\n\nTesting list with: alpha:");
    LLNode<String> temp1;
    temp1 = new LLNode<String>("alpha");
    
    LLNode<String> singletonList = temp1;
    
    System.out.println("\n  recursive print: "); recPrintList(singletonList); 
    System.out.println("\n  iterative print: "); iterPrintList(singletonList);
    System.out.println("\n  recursive size: ");  System.out.println(recListSize(singletonList)); 
    System.out.println("\n  iterative size: "); System.out.println(iterListSize(singletonList)); 
    
    System.out.println("\n  adding \"final\" ... ");
    singletonList = recInsertEnd("final", singletonList);
    System.out.println("\n  recursive print: "); recPrintList(singletonList); 
     
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
    
    System.out.println("\n  recursive print: "); recPrintList(multiList); 
    System.out.println("\n  iterative print: "); iterPrintList(multiList);
    System.out.println("\n  recursive size: ");  System.out.println(recListSize(multiList)); 
    System.out.println("\n  iterative size: "); System.out.println(iterListSize(multiList)); 
    
    System.out.println("\n  adding \"final\" ... ");
    multiList = recInsertEnd("final", multiList);
    System.out.println("\n  recursive print: "); recPrintList(multiList); 
  }
}
