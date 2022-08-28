//---------------------------------------------------------------------------
// LargeIntList.java           by Dale/Joyce/Weems                  Chapter 6
// 
// A specialized list to support Large Integer ADT
//---------------------------------------------------------------------------
package ch06.largeInts;

import support.DLLNode;
import java.util.Iterator;
 
public class LargeIntList
{
  protected DLLNode<Byte> listFirst;   // Ref to the first node on list
  protected DLLNode<Byte> listLast;    // Ref to the last node on the list
  protected int numElements;           // Number of elements in the list
 
  public LargeIntList()
  // Creates an empty list object
  {
    numElements = 0;
    listFirst = null;
    listLast = null;
  }

  public int size()
  // Returns the number of elements on this list.
  {
    return numElements;
  }

  public Iterator<Byte> forward()
  // Returns an Iterator that iterates from front to rear.
  {
    return new Iterator<Byte>()
    {
      private DLLNode<Byte> next = listFirst; // next node to return

      public boolean hasNext()
      // Returns true if the iteration has more elements; otherwise false.
      {
        return (next != null);
      }
      
      public Byte next()
      // Returns the next element in the iteration.
      // Throws NoSuchElementException - if no more elements
      { 
        if (!hasNext())
          throw new IndexOutOfBoundsException("Illegal invocation of " +
                           " next in LargeIntList forward iterator.\n");
  
        Byte hold = next.getInfo();        // holds info for return
        next = next.getForward();
        return hold;
      }

      public void remove()
      // Throws UnsupportedOperationException.
      {
        throw new UnsupportedOperationException("Unsupported remove " +
                        "attempted on LargeIntList forward iterator.");
      }
    };
  }

  public Iterator<Byte> reverse()
  // Returns an Iterator that iterates rear to front.
  {
    return new Iterator<Byte>()
    {
      private DLLNode<Byte> next = listLast; // next node to return

      public boolean hasNext()
      // Returns true if the iteration has more elements; otherwise false.
      {
        return (next != null);
      }
      
      public Byte next()
      // Returns the next element in the iteration.
      // Throws NoSuchElementException - if no more elements
      { 
        if (!hasNext())
          throw new IndexOutOfBoundsException("Illegal invocation of " +
                            "next in LargeIntList reverse iterator.\n");
  
        Byte hold = next.getInfo();        // holds info for return
        next = next.getBack();
        return hold;
      }

      public void remove()
      // Throws UnsupportedOperationException.
      {
        throw new UnsupportedOperationException("Unsupported remove " +
                        "attempted on LargeIntList forward iterator.");
      }
    };
  }

  public void addFront (byte element)
  // Adds the value of element to the beginning of this list
  {
    DLLNode<Byte> newNode = new DLLNode<Byte>(element); // node being added
    newNode.setForward(listFirst);
    newNode.setBack(null);
    if (listFirst == null)   // Adding into an empty list
    {
      listFirst = newNode;
      listLast = newNode;
    }
    else                     // Adding into a non-empty list
    {
      listFirst.setBack(newNode);
      listFirst = newNode;
    }
    numElements++;
  }

  public void addEnd (byte element)
  // Adds the value of element to the end of this list
  {
    DLLNode<Byte> newNode = new DLLNode<Byte>(element); // node being added
    newNode.setForward(null);
    newNode.setBack(listLast);
    if (listFirst == null)    // Adding into an empty list
    {
      listFirst = newNode;
      listLast = newNode;
    }
    else                      // Adding into a non-empty list
    {
      listLast.setForward(newNode);
      listLast = newNode;
    }
    numElements++;
  }
}