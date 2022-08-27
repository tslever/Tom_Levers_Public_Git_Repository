//----------------------------------------------------------------
// ABListIterator.java        by Dale/Joyce/Weems        Chapter 6
//
// Example of an "external class" iterator for the ABList class
//----------------------------------------------------------------
package ch06.lists;

import java.util.Iterator;

public class ABListIterator<T> implements Iterator<T>
{
  private ABList<T> theList;
  private int previousPos = -1;

  public ABListIterator(ABList<T> target) 
  {
    theList = target;
  }

  public boolean hasNext()
  // Returns true if the iteration has more elements; otherwise returns false.
  {
    return (previousPos < (theList.size() - 1)) ;
  }
      
  public T next()
  // Returns the next element in the iteration.
  // Throws NoSuchElementException - if the iteration has no more elements
  { 
    if (!hasNext())
      throw new IndexOutOfBoundsException("Illegal invocation of next " + 
                         " in ABList iterator.\n");
    previousPos++;
    return theList.elements[previousPos];
  }

  public void remove()
  // Removes from the underlying list the last element returned by this iterator. 
  // This method should be called only once per call to next(). 
  // The behavior of an iterator is unspecified if the underlying list is modified 
  // while the iteration is in progress in any way other than by calling this method.
  {
    for (int i = previousPos; i <= theList.numElements - 2; i++)
      theList.elements[i] = theList.elements[i+1];
    theList.elements[theList.numElements - 1] = null;
    theList.numElements--;
    previousPos--;  
  }
}
