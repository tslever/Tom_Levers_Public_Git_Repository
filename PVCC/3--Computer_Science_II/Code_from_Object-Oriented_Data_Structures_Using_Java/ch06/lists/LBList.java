//----------------------------------------------------------------------------
// LBList.java                by Dale/Joyce/Weems                    Chapter 6
// Link-Based List
//
// Null elements are not permitted on a list. The list is unbounded.
//----------------------------------------------------------------------------

package ch06.lists;

import java.util.Iterator;
import support.LLNode;

public class LBList<T> implements ListInterface<T>  
{
  protected LLNode<T> front;     // reference to the front of this list
  protected LLNode<T> rear;      // reference to the rear of this list
  protected int numElements = 0; // number of elements in this list

  // set by find method
  protected boolean found;        // true if target found, else false
  protected int targetIndex;      // list index of target, if found
  protected LLNode<T> location;   // node containing target, if found
  protected LLNode<T> previous;   // node preceding location

  public LBList()
  {
    numElements = 0;
    front = null;
    rear = null;
  }

  public boolean add(T element)
  // Adds element to the end of this list.
  {
    LLNode<T> newNode = new LLNode<T>(element);
    if (rear == null)
      front = newNode;
    else
      rear.setLink(newNode);
    rear = newNode;
    numElements++;
    return true;
  }

  protected void find(T target)
  // Searches list for an occurence of an element e such that
  // e.equals(target). If successful, sets instance variables
  // found to true, location to node containing e, previous
  // to the node that links to location, and index to the index of
  // the location. If not successful, sets found to false.
  {
    location = front;
    found = false;
    targetIndex = -1;

    while (location != null) 
    {
      targetIndex++;
      if (location.getInfo().equals(target))  // if they match
      {
       found = true;
       return;
      }
      else
      {
        previous = location;
        location = location.getLink();
      }
    }
  }

  public int size()
  // Returns the number of elements on this list. 
  {
    return numElements;
  }

  public boolean contains (T target)
  // Returns true if this list contains an element e such that 
  // e.equals(target); otherwise, returns false.
  {
    find(target);
    return found;
  }

  public boolean remove (T target)
  // Removes an element e from this list such that e.equals(target)
  // and returns true; if no such element exists, returns false.
  {
    find(target);
    if (found)
    {
      if (front == location)     
        front = front.getLink();    // remove first node
      else
        previous.setLink(location.getLink());  // remove node at location

      if (front == null) rear = null;   // removed only element

      numElements--;
    }
    return found;
  }

  public T get(T target)
  // Returns an element e from this list such that e.equals(target);
  // if no such element exists, returns null.
  {
    find(target);    
    if (found)
      return location.getInfo();
    else
      return null;
  }
    
  public boolean isEmpty()
  // Returns true if this list is empty; otherwise, returns false.
  {
    return (numElements == 0);  
  }

  public boolean isFull()
  // Returns false.
  {
    return false;  // list is unbounded.
  } 
   
  public void add(int index, T element)
  // Throws IndexOutOfBoundsException if passed an index argument
  // such that index < 0 or index > size().
  // Otherwise, adds element to this list at position index; all current 
  // elements at that index or higher have 1 added to their index.
  {
    if ((index < 0) || (index > size()))
      throw new IndexOutOfBoundsException("Illegal index of " + index + 
                             " passed to LBList add method.\n");

    LLNode<T> newNode = new LLNode<T>(element);

    if (index == 0) // add to front
    {
      if (front == null) // adding to empty list
      { 
        front = newNode; rear = newNode;
      }
      else
      {
        newNode.setLink(front);
        front = newNode;
      }
    }
    else
    if (index == size()) // add to rear
    {
      rear.setLink(newNode);
      rear = newNode;
    }
    else  // add in interior part of list
    {
      LLNode<T> node = front;                                   
      for (int i = 0; i < (index - 1); i++)
         node = node.getLink();
      newNode.setLink(node.getLink());
      node.setLink(newNode);
    }
    numElements++;
  }
  
  public T set(int index, T newElement)
  // Throws IndexOutOfBoundsException if passed an index argument
  // such that index < 0 or index >= size().
  // Otherwise, replaces element on this list at position index with
  // newElement and returns the replaced element.
  {
    if ((index < 0) || (index >= size()))
      throw new IndexOutOfBoundsException("Illegal index of " + index + 
                             " passed to LBList set method.\n");

    LLNode<T> node = front;                                   
    for (int i = 0; i < index; i++)
         node = node.getLink();
    T hold = node.getInfo();
    node.setInfo(newElement);
    return hold;
  }
    
  public T get(int index)
  // Throws IndexOutOfBoundsException if passed an index argument
  // such that index < 0 or index >= size().
  // Otherwise, returns the element on this list at position index.
  {
    if ((index < 0) || (index >= size()))
      throw new IndexOutOfBoundsException("Illegal index of " + index + 
                             " passed to LBList set method.\n");

    LLNode<T> node = front;                                   
    for (int i = 0; i < index; i++)
         node = node.getLink();
    return node.getInfo();
  }  

  public int indexOf(T target)
  // If this list contains an element e such that e.equals(target), 
  // then returns the index of the first such element.
  // Otherwise, returns -1.
  {
    find(target);    
    if (found)
      return targetIndex;
    else  
      return -1;
  }
  
  public T remove(int index)
  // Throws IndexOutOfBoundsException if passed an index argument
  // such that index < 0 or index >= size().
  // Otherwise, removes element on this list at position index and
  // returns the removed element; all current elements at positions
  // higher than that index have 1 subtracted from their position.
  {
    if ((index < 0) || (index >= size()))
      throw new IndexOutOfBoundsException("Illegal index of " + index + 
                             " passed to LBList remove method.\n");

    T hold;  // holds info for return
    if (index == 0) // remove the front node
    {
      hold = front.getInfo();
      front = front.getLink();
      if (numElements == 1)   // removing only node
        rear = null;
    }
    else
    {
      // locate previous node
      LLNode<T> node = front;                                   
      for (int i = 0; i < (index - 1); i++)
         node = node.getLink();
      hold = node.getLink().getInfo();
      if (node.getLink() == rear)  // removing rear node
        rear = node;
      node.setLink(node.getLink().getLink());
    }

    numElements--;
    return hold;
  }

  public Iterator<T> iterator()
  // Returns an Iterator over this list.
  {
    return new Iterator<T>()
    {
      private LLNode<T> prevPos = null;  // node before node just returned
      private LLNode<T> currPos = null;  // node just returned
      private LLNode<T> nextPos = front; // next node to return

      public boolean hasNext()
      // Returns true if the iteration has more elements; otherwise returns false.
      {
        return (nextPos != null);
      }
      
      public T next()
      // Returns the next element in the iteration.
      // Throws NoSuchElementException - if the iteration has no more elements
      { 
        if (!hasNext())
          throw new IndexOutOfBoundsException("Illegal invocation of next " + 
                             " in LBList iterator.\n");
  
        T hold = nextPos.getInfo();              // holds info for return
        if (currPos != null) prevPos = currPos;  // in case element was removed
        currPos = nextPos;
        nextPos = nextPos.getLink();
        return hold;
      }

      public void remove()
      // Removes from the underlying representation the last element returned
      // by this iterator. This method should be called only once per call to
      // next(). The behavior of an iterator is unspecified if the underlying
      // representation is modified while the iteration is in progress in any 
      // way other than by calling this method.
      {
        if (currPos == null)  // there is no last element returned by the iterator
          return;
        else
        {
          if (prevPos == null)  // removing front element
          {
            front = nextPos;
            currPos = null;
            if (front == null) // removed only element
              rear = null;
          }
          else
          {
            prevPos.setLink(nextPos);
            currPos = null;
          }
          numElements--;
        }
      }
    };
  }
}
