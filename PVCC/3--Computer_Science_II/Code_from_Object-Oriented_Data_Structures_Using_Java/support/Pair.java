//---------------------------------------------------------------------
// Pair.java            by Dale/Joyce/Weems                   Chapter 5
//
// Example of a class that requires client applications to use 
// Comparable class as generic argument.
//---------------------------------------------------------------------

package support;

public class Pair<T extends Comparable<T>>
{
  protected T first;
  protected T second;

  public Pair(T f, T s)
  {
    first = f; second = s;
  }

  public T larger()
  {
    if (first.compareTo(second) > 0)
       return first;
    else
       return second;
  }
}
 