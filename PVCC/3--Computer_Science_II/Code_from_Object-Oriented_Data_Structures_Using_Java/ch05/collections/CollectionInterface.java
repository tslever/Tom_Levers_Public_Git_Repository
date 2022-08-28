//----------------------------------------------------------------------------
// CollectionInterface.java        by Dale/Joyce/Weems               Chapter 5
//
// Interface for a class that implements a collection of T.
// A collection allows addition, removal, and access of elements.
//
// Nullelements are not allowed. Duplicate elements are allowed.
//----------------------------------------------------------------------------

package ch05.collections;

public interface CollectionInterface<T>
{
  boolean add(T element);
  // Attempts to add element to this collection.
  // Returns true if successful, false otherwise.

  T get(T target);
  // Returns an element e from this collection such that e.equals(target).
  // If no such e exists, returns null.

  boolean contains(T target);
  // Returns true if this collection contains an element e such that
  // e.equals(target); otherwise returns false.

  boolean remove (T target);
  // Removes an element e from this collection such that e.equals(target)
  // and returns true. If no such e exists, returns false.

  boolean isFull();
  // Returns true if this collection is full; otherwise, returns false.

  boolean isEmpty();
  // Returns true if this collection is empty; otherwise, returns false.
  
  int size();
  // Returns the number of elements in this collection.
}
