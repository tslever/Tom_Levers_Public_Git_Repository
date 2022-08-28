//----------------------------------------------------------------------
// ArrayListStack.java        by Dale/Joyce/Weems              Chapter 2
//
// Implements an unbounded stack using an ArrayList.
//----------------------------------------------------------------------

package ch02.stacks;

import java.util.ArrayList; 

public class ArrayListStack<T> implements StackInterface<T>
{
  protected ArrayList<T> elements; // ArrayList that holds stack elements

  public ArrayListStack() 
  {
    elements = new ArrayList<T>();      
  }

  public void push(T element)   
  // Places element at the top of this stack.
  {
    elements.add(element);
  }

  public void pop()               
  // Throws StackUnderflowException if this stack is empty,
  // otherwise removes top element from this stack.
  {
    if (isEmpty())
      throw new StackUnderflowException("Pop attempted on an empty stack.");
    else 
      elements.remove(elements.size() - 1);
  }

  public T top()             
  // Throws StackUnderflowException if this stack is empty,
  // otherwise returns top element of this stack.
  {
    T topOfStack = null;
    if (isEmpty())
      throw new StackUnderflowException("Top attempted on an empty stack.");    
    else 
      topOfStack = elements.get(elements.size() - 1);
    return topOfStack;
  }

  public boolean isEmpty()         
  // Returns true if this stack is empty, otherwise returns false.
  {
    return (elements.size() == 0);
  }
  
  public boolean isFull()
  // Returns false - an ArrayList stack is never full.
  {              
    return false;
  }

}