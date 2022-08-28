//----------------------------------------------------------------------------
// Heap.java               by Dale/Joyce/Weems                       Chapter 9
// Priority Queue using Heap (implemented with an ArrayList)
//
// Two constructors are provided: one that use the natural order of the 
// elements as defined by their compareTo method and one that uses an 
// ordering based on a comparator argument. 
//
// Null elements are not allowed. Duplicate elements are allowed.
//----------------------------------------------------------------------------
package ch09.priorityQueues;

import java.util.*; // ArrayList, Comparator

public class HeapPriQ<T> implements PriQueueInterface<T>
{
  protected ArrayList<T> elements;  // priority queue elements
  protected int lastIndex;          // index of last element in priority queue
  protected int maxIndex;           // index of last position in ArrayList

  protected Comparator<T> comp;

  public HeapPriQ(int maxSize)
  // Precondition: T implements Comparable
  {
    elements = new ArrayList<T>(maxSize);
    lastIndex = -1;
    maxIndex = maxSize - 1;

    comp = new Comparator<T>()
    {
       public int compare(T element1, T element2)
       {
         return ((Comparable)element1).compareTo(element2);
       }
    };
  }

  public HeapPriQ(int maxSize, Comparator<T> comp)
  // Precondition: T implements Comparable
  {
    elements = new ArrayList<T>(maxSize);
    lastIndex = -1;
    maxIndex = maxSize - 1;

    this.comp = comp;
  }

  public boolean isEmpty()
  // Returns true if this priority queue is empty; otherwise, returns false.
  {
    return (lastIndex == -1);
  }

  public boolean isFull()
  // Returns true if this priority queue is full; otherwise, returns false.
  {
    return (lastIndex == maxIndex);
  }
  
  public int size()
  // Returns the number of elements on this priority queue. 
  {
    return lastIndex + 1;
  }

  private void reheapUp(T element)
  // Current lastIndex position is empty.
  // Inserts element into the tree and ensures shape and order properties.
  {
    int hole = lastIndex;
    while ((hole > 0)    // hole is not root and element > hole's parent
           &&                                               
      (comp.compare(element, elements.get((hole - 1) / 2)) > 0)) 
      {
      // move hole's parent down and then move hole up
      elements.set(hole,elements.get((hole - 1) / 2)); 
      hole = (hole - 1) / 2;                                
    }
    elements.set(hole, element);  // place element into final hole
  }

  public void enqueue(T element) throws PriQOverflowException
  // Throws PriQOverflowException if this priority queue is full;
  // otherwise, adds element to this priority queue.
  {
    if (lastIndex == maxIndex)
      throw new PriQOverflowException("Priority queue is full");
    else
    {
      lastIndex++;
      elements.add(lastIndex,element);
      reheapUp(element);
    }
  }

  private int newHole(int hole, T element)
  // If either child of hole is larger than element, return the index
  // of the larger child; otherwise, return the index of hole.
  {
    int left = (hole * 2) + 1;
    int right = (hole * 2) + 2;

    if (left > lastIndex)
      // hole has no children
      return hole;         
    else
    if (left == lastIndex)
      // hole has left child only
      if (comp.compare(element, elements.get(left)) < 0)             
        // element < left child
        return left;
      else
        // element >= left child
        return hole;
    else
    // hole has two children 
    if (comp.compare(elements.get(left), elements.get(right)) < 0)
      // left child < right child
      if (comp.compare(elements.get(right), element) <= 0)
        // right child <= element
        return hole;
      else
        // element < right child
        return right;
    else
    // left child >= right child
    if (comp.compare(elements.get(left), element) <= 0)
      // left child <= element
      return hole;
    else
      // element < left child
      return left;
  }

  private void reheapDown(T element)
  // Current root position is "empty";
  // Inserts element into the tree and ensures shape and order properties.
  {
    int hole = 0;      // current index of hole
    int next;          // next index where hole should move to

    next = newHole(hole, element);   // find next hole
    while (next != hole)
    {
      elements.set(hole,elements.get(next));  // move element up
      hole = next;                            // move hole down
      next = newHole(hole, element);          // find next hole
    }
    elements.set(hole, element);              // fill in the final hole
  }

  public T dequeue() throws PriQUnderflowException
  // Throws PriQUnderflowException if this priority queue is empty;
  // otherwise, removes element with highest priority from this 
  // priority queue and returns it.
  {
    T hold;      // element to be dequeued and returned
    T toMove;    // element to move down heap

    if (lastIndex == -1)
      throw new PriQUnderflowException("Priority queue is empty");
    else
    {
      hold = elements.get(0);              // remember element to be returned
      toMove = elements.remove(lastIndex); // element to reheap down
      lastIndex--;                         // decrease priority queue size
      if (lastIndex != -1)                 // if priority queue is not empty
         reheapDown(toMove);                  // restore heap properties
      return hold;                         // return largest element
    }
  }

  @Override
  public String toString()
  // Returns a string of all the heap elements.
  {
    String theHeap = new String("the heap is:\n");
    for (int index = 0; index <= lastIndex; index++)
      theHeap = theHeap + index + ". " + elements.get(index) + "\n";
    return theHeap;
  }
}