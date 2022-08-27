//----------------------------------------------------------------------------
// ITDArrayBoundedQueue.java         by Dale/Joyce/Weems             Chapter 4
//
// Interactive Test Driver for the ArrayBoundedQueue class
//----------------------------------------------------------------------------
package ch04.queues;

import java.util.*;

public class ITDArrayBoundedQueue
{
  public static void main(String[] args)
  { 
    QueueInterface<String> test = new ArrayBoundedQueue<String>();
    Scanner scan = new Scanner(System.in);

    String skip;       // skip end of line after reading an integer
    boolean keepGoing; // flag for "choose operation" loop
    int constructor;   // indicates user's choice of constructor
    int operation;     // indicates user's choice of operation

    String enqueueString = "", dequeueString = "";  // used by operations

    // Handle test name  
    System.out.println("What is the name of this test?");
    String testName = scan.nextLine();
    System.out.println("\nThis is test " + testName + "\n");
    
    // Handle constructor
    System.out.println("Choose a constructor:");
    System.out.println("1: ArrayBoundedQueue( )");
    System.out.println("2: ArrayBoundedQueue(int maxSize)");
    if (scan.hasNextInt())
      constructor = scan.nextInt();
    else
    {
      System.out.println("Error: you must enter an integer.");
      System.out.println("Terminating test.");
      return;
    }
    skip = scan.nextLine();

    switch (constructor)
    {
      case 1:
        test = new ArrayBoundedQueue<String>();
        break;
        
      case 2:
        System.out.println("Enter a maximum size:");
        int maxSize;
        if (scan.hasNextInt())
          maxSize = scan.nextInt();
        else
        {
          System.out.println("Error: you must enter an integer.");
          System.out.println("Terminating test.");
          return;
        }
        skip = scan.nextLine();
        test = new ArrayBoundedQueue<String>(maxSize);
        break;
        
      default:
        System.out.println("Error in constructor choice. Terminating test.");
        return;
    }
    
    // Handle test cases
    keepGoing = true;  
    while (keepGoing)
    {
      System.out.println("\nChoose an operation:");
      System.out.println("1: enqueue(element)");
      System.out.println("2: String dequeue()");
      System.out.println("3: boolean isFull()");
      System.out.println("4: boolean isEmpty()");
      System.out.println("5: int size()");      
      System.out.println("6: stop Testing");
      if (scan.hasNextInt())
        operation = scan.nextInt();
      else
      {
        System.out.println("Error: you must enter an integer.");
        System.out.println("Terminating test.");
        return;
      } 
      skip = scan.nextLine();

      switch (operation)
      {
        case 1:  // enqueue
        System.out.println("Enter string to enqueue:");
        enqueueString = scan.nextLine();
        System.out.println("enqueue(\"" + enqueueString + "\")");
        try
        { 
          test.enqueue(enqueueString);
        }
        catch (QueueOverflowException QOFException)
        {
          System.out.println("Overflow Exception: " + QOFException.getMessage());
        }        
        break;
        
        case 2:  // dequeue
        System.out.println("dequeue()");
        try
        { 
          dequeueString = test.dequeue();
        }
        catch (QueueUnderflowException QUFException)
        {
          System.out.println("Underflow Exception: " + QUFException.getMessage());
          break;
        }        
        System.out.println("Result: " + dequeueString + " was returned.");
        break;
        
        case 3:  // isFull
        System.out.println("isFull()");
        System.out.println("Result: " + test.isFull());
        break;
        
        case 4:  // isEmpty
        System.out.println("isEmpty()");
        System.out.println("Result: " + test.isEmpty());
        break;
        
        case 5:  // size
        System.out.println("size()");
        System.out.println("Result: " + test.size());
        break;
        
        case 6:  // stop testing
        keepGoing = false;
        break;
        
        default:
        System.out.println("Error in operation choice. Terminating test.");
        return;
      }
    }

  System.out.println("End of Interactive Test Driver");
  }
}