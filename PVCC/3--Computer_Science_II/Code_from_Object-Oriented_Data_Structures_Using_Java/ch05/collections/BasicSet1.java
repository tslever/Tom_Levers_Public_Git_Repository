//---------------------------------------------------------------------------
// BasicSet1.java              by Dale/Joyce/Weems                  Chapter 5
//
// Implements the CollectionInterface by extending LinkedCollection.
// Overrides add method so that duplicate elements are not added.
//
// Null elements are not allowed.
// One constructor is provided, one that creates an empty collection.
//---------------------------------------------------------------------------
package ch05.collections;

public class BasicSet1<T> extends LinkedCollection<T> 
                          implements CollectionInterface<T>  
{
  public BasicSet1() 
  {
    super();
  }
  
  @Override
  public boolean add(T element)
  // If element is not already contained in this collection adds element to 
  // this collection and returns true; otherwise returns false.
  {
    if (!this.contains(element))
    {
      super.add(element);
      return true;
    }
    else
      return false;
  }
}
