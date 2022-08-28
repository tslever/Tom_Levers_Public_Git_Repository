//---------------------------------------------------------------------------
// MapEntry.java              by Dale/Joyce/Weems                   Chapter 8
//
// Provides key, value pairs for use with a Map.
// Keys are immutable.
//---------------------------------------------------------------------------
package ch08.maps;

public class MapEntry<K, V>
{
  protected K key;
  protected V value;
  
  MapEntry(K k, V v)
  {
    key = k; value = v;
  }
  
  public K getKey()  {return key;}
  public V getValue(){return value;}
  public void setValue(V v){value = v;}

  @Override
  public String toString()
  // Returns a string representing this MapEntry.
  {
    return "Key  : " + key + "\nValue: " + value;
  }
}
 