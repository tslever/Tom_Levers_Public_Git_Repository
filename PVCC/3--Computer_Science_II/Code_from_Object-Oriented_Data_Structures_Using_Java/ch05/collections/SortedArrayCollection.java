//----------------------------------------------------------------------------
// SortedArrayCollection.java        by Dale/Joyce/Weems             Chapter 5
//
// Implements the CollectionInterface using a sorted array. The collection is
// stored in increasing order as defined by the compareTo method of the added 
// elements. Only Comparable elements should be added to a collection. It is
// assumed that the compareTo method and equals method as defined for added
// elements are consistent.
//
// Null elements are not permitted in this collection.
//
// Two constructors are provided: one that creates a collection of a default
// original capacity, and one that allows the calling program to specify the 
// original capacity. The collection is unbounded.
//----------------------------------------------------------------------------

package ch05.collections;

public class SortedArrayCollection<T> implements CollectionInterface<T>  
{
  protected final int DEFCAP = 100; // default capacity
  protected int origCap;            // original capacity
  protected T[] elements;           // array to hold collection elements
  protected int numElements = 0;    // number of elements in this collection

  // set by find method
  protected boolean found;  // true if target found, otherwise false
  protected int location;   // indicates location of target if found,
                            // indicates add index if not found

  public SortedArrayCollection() 
  {
    elements = (T[]) new Object[DEFCAP];
    origCap = DEFCAP;
  }

  public SortedArrayCollection(int capacity) 
  {
    elements = (T[]) new Object[capacity];
    this.origCap = capacity;
  }

  protected void enlarge()
  // Increments the capacity of the collection by an amount 
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
  // Searches elements for an occurrence of an element e such that
  // e.equals(target). If successful, sets instance variables
  // found to true and location to the array index of e. If
  // not successful, sets found to false and location to the 
  // array index where target should be inserted.
  {
    location = 0;
    found = false;
    if (!isEmpty())
       recFind(target, 0, numElements - 1);
  }

  protected void recFind(T target, int first, int last)
  // Used by find.
  {
    int result;       // result of the comparison
    if (first > last)
    {
      found = false;
      result = ((Comparable)target).compareTo(elements[location]);
      if (result > 0)
         location++;    // adjust location to indicate insert index
    }
    else
    {
      location = (first + last) / 2;
      result = ((Comparable)target).compareTo(elements[location]);
      if (result == 0)  // found target
        found = true;
      else
      if (result > 0)   // target too high
        recFind(target, location + 1, last);
      else               // target too low
        recFind(target, first, location - 1);
     }
  }

  public boolean add(T element)
  // Precondition:  element is Comparable to previously added objects
  //  
  // Adds element to this collection.
  {
    if (numElements == elements.length)
      enlarge();

    find(element); // sets location to index where element belongs
    
    for (int index = numElements; index > location; index--)
      elements[index] = elements[index - 1];

    elements[location] = element;
    numElements++;
    return true;
  }

  public boolean remove (T target)
  // Removes an element e from this collection such that e.equals(target)
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
  // Returns the number of elements on this collection. 
  {
    return numElements;
  }

  public boolean contains (T target)
  // Returns true if this collection contains an element e such that 
  // e.equals(target); otherwise, returns false.
  {
    find(target);
    return found;
  }

  public T get(T target)
  // Returns an element e from this collection such that e.equals(target);
  // if no such element exists, returns null.
  {
    find(target);    
    if (found)
      return elements[location];
    else
      return null;
  }
  
  public boolean isEmpty()
  // Returns true if this collection is empty; otherwise, returns false.
  {
    return (numElements == 0);  
  }

  public boolean isFull()
  // This collection is unbounded so always returns false.
  {
    return false;
  }
}
