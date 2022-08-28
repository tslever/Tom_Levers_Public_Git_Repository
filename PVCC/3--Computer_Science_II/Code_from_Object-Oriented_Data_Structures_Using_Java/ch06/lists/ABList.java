//----------------------------------------------------------------------------
// ABList.java                by Dale/Joyce/Weems                    Chapter 6
// Array-Based List
//
// Null elements are not permitted on a list. The list is unbounded.
//
// Two constructors are provided: one that creates a list of a default
// original capacity, and one that allows the calling program to specify the 
// original capacity.
//----------------------------------------------------------------------------

package ch06.lists;

import java.util.Iterator;

public class ABList<T> implements ListInterface<T>  
{
  protected final int DEFCAP = 100; // default capacity
  protected int origCap;            // original capacity
  protected T[] elements;           // array to hold this list’s elements
  protected int numElements = 0;    // number of elements in this list

  // set by find method
  protected boolean found;  // true if target found, otherwise false
  protected int location;   // indicates location of target if found

  public ABList() 
  {
    elements = (T[]) new Object[DEFCAP];
    origCap = DEFCAP;
  }

  public ABList(int origCap) 
  {
    elements = (T[]) new Object[origCap];
    this.origCap = origCap;
  }

  protected void enlarge()
  // Increments the capacity of the list by an amount 
  // equal to the original capacity.
  {
    // Create the larger array.
    T[] larger = (T[]) new Object[elements.length + origCap];
    
    // Copy the contents from the smaller array into the larger array.
    for (int i = 0; i < numElements; i++)
    {
      larger[i] = elements[i];
    }
    
    // Reassign elements reference.
    elements = larger;
  }

  protected void find(T target)
  // Searches list for an occurence of an element e such that
  // e.equals(target). If successful, sets instance variables
  // found to true and location to the array index of e. If
  // not successful, sets found to false.
  {
    location = 0;
    found = false;

    while (location < numElements) 
    {
      if (elements[location].equals(target))
      {  
        found = true;
        return;
      }
      else
        location++;
    }
  }

  public boolean add(T element)
  // Adds element to end of this list.
  {
    if (numElements == elements.length)
      enlarge();
    elements[numElements] = element;
    numElements++;
    return true;
  }

  public boolean remove (T target)
  // Removes an element e from this list such that e.equals(target)
  // and returns true; if no such element exists, returns false.
  {
    find(target);    
    if (found)
    {
      for (int i = location; i <= numElements - 2; i++)
        elements[i] = elements[i+1];
      elements[numElements - 1] = null;
      numElements--;  
    }
    return found;
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

  public T get(T target)
  // Returns an element e from this list such that e.equals(target);
  // if no such element exists, returns null.
  {
    find(target);    
    if (found)
      return elements[location];
    else
      return null;
  }

  public boolean isEmpty()
  // Returns true if this list is empty; otherwise, returns false.
  {
    return (numElements == 0);  
  }

  public boolean isFull()
  // Returns false - the list is unbounded.
  {
    return false;
  }

  public void add(int index, T element)
  // Throws IndexOutOfBoundsException if passed an index argument
  // such that index < 0 or index > size().
  // Otherwise, adds element to this list at position index; all current 
  // elements at that index or higher have 1 added to their index.
  {
    if ((index < 0) || (index > size()))
      throw new IndexOutOfBoundsException("Illegal index of " + index + 
                             " passed to ABList add method.\n");
                                    
    if (numElements == elements.length)
      enlarge();
    
    for (int i = numElements; i > index; i--)
      elements[i] = elements[i - 1];

    elements[index] = element;
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
                             " passed to ABList set method.\n");
 
    T hold = elements[index];
    elements[index] = newElement;
    return hold;
  }
    
  public T get(int index)
  // Throws IndexOutOfBoundsException if passed an index argument
  // such that index < 0 or index >= size().
  // Otherwise, returns the element on this list at position index.
  {
    if ((index < 0) || (index >= size()))
      throw new IndexOutOfBoundsException("Illegal index of " + index + 
                             " passed to ABList get method.\n");
 
    return elements[index];
  }  

  public int indexOf(T target)
  // If this list contains an element e such that e.equals(target), 
  // then returns the index of the first such element.
  // Otherwise, returns -1.
  {
    find(target);
    if (found)
      return location;
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
                             " passed to ABList remove method.\n");

    T hold = elements[index];
    for (int i = index; i < numElements-1; i++)
      elements[i] = elements[i + 1];
    elements[numElements-1] = null;
    numElements--;
    return hold;
  }

  public Iterator<T> iterator()
  // Returns an Iterator over this list.
  {
    return new Iterator<T>()
    {
      private int previousPos = -1;

      public boolean hasNext()
      // Returns true if the iteration has more elements; otherwise returns false.
      {
        return (previousPos < (size() - 1)) ;
      }
      
      public T next()
      // Returns the next element in the iteration.
      // Throws NoSuchElementException - if the iteration has no more elements
      { 
        if (!hasNext())
          throw new IndexOutOfBoundsException("Illegal invocation of next " + 
                             " in ABList iterator.\n");
        previousPos++;
        return elements[previousPos];
      }

      public void remove()
      // Removes from the underlying representation the last element returned
      // by this iterator. This method should be called only once per call to
      // next(). The behavior of an iterator is unspecified if the underlying
      // representation is modified while the iteration is in progress in any 
      // way other than by calling this method.
      {
        for (int i = previousPos; i <= numElements - 2; i++)
          elements[i] = elements[i+1];
        elements[numElements - 1] = null;
        numElements--;
        previousPos--;  
      }
    };
  }
}
