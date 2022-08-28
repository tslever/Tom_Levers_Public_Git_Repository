//---------------------------------------------------------------------------
// BasicSet2.java             by Dale/Joyce/Weems                   Chapter 5
//
// Implements the CollectionInterface by wrapping a LinkedCollection.
// Ensures that duplicate elements are not added.
//
// Null elements are not allowed.
// One constructor is provided, one that creates an empty collection.
//---------------------------------------------------------------------------
package ch05.collections;

public class BasicSet2<T> implements CollectionInterface<T>  
{
  LinkedCollection<T> set;
  
  public BasicSet2() 
  {
    set = new LinkedCollection<T>();
  }

  public boolean add(T element)
  // If element is not already contained in this collection adds element to 
  // this collection and returns true; otherwise returns false.
  {
    if (!this.contains(element))
    {
      set.add(element);
      return true;
    }
    else
      return false;
  }
  
  public int size(){return set.size();}
  public boolean contains (T target){return set.contains(target);}
  public boolean remove (T target){return set.remove(target);}
  public T get(T target){return set.get(target);}
  public boolean isEmpty(){return set.isEmpty();}
  public boolean isFull(){return set.isFull();}
}
