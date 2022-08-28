package ch02.stacks;

public class ArrayBoundedStack01<T> implements StackInterface<T> 
{
  public ArrayBoundedStack01() 
  {     }

  public void push(T element)
  // Throws StackOverflowException if this stack is full,
  // otherwise places element at the top of this stack.
  {    }

  public void pop()
  // Throws StackUnderflowException if this stack is empty,
  // otherwise removes top element from this stack.
  {    }

  public T top()
  // Throws StackUnderflowException if this stack is empty,
  // otherwise returns top element of this stack.
  {                 
    return null;
  }

  public boolean isEmpty()
  // Returns true if this stack is empty, otherwise returns false.
  {              
    return true;
  }

  public boolean isFull()
  // Returns true if this stack is full, otherwise returns false.
  {              
    return false;
  }
}