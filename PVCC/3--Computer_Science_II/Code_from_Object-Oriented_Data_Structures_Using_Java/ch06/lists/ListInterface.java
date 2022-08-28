//----------------------------------------------------------------------------
// ListInterface.java            by Dale/Joyce/Weems                 Chapter 6
//
// Lists are unbounded and allow duplicate elements, but do not allow 
// null elements. As a general precondition, null elements are not passed as 
// arguments to any of the methods.
//
// During an iteration through the list the only change that can safely be
// made to the list is through the remove method of the iterator.
//----------------------------------------------------------------------------

package ch06.lists;

import java.util.*;
import ch05.collections.CollectionInterface;

public interface ListInterface<T> extends CollectionInterface<T>, Iterable<T>
{
  void add(int index, T element);
  // Throws IndexOutOfBoundsException if passed an index argument
  // such that index < 0 or index > size().
  // Otherwise, adds element to this list at position index; all current 
  // elements at that position or higher have 1 added to their index.
  // Optional. Throws UnsupportedOperationException if not supported.
  
  T set(int index, T newElement);
  // Throws IndexOutOfBoundsException if passed an index argument
  // such that index < 0 or index >= size().
  // Otherwise, replaces element on this list at position index with
  // newElement and returns the replaced element.
  // Optional. Throws UnsupportedOperationException if not supported.
    
  T get(int index);
  // Throws IndexOutOfBoundsException if passed an index argument
  // such that index < 0 or index >= size().
  // Otherwise, returns the element on this list at position index.
  
  int indexOf(T target);
  // If this list contains an element e such that e.equals(target), 
  // then returns the index of the first such element.
  // Otherwise, returns -1.
  
  T remove(int index);
  // Throws IndexOutOfBoundsException if passed an index argument
  // such that index < 0 or index >= size().
  // Otherwise, removes element on this list at position index and
  // returns the removed element; all current elements at positions
  // higher than index have 1 subtracted from their position.
}
