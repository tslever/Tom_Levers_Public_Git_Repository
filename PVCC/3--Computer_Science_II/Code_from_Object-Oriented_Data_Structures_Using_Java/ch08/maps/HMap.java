//---------------------------------------------------------------------------
// HMap.java                by Dale/Joyce/Weems                     Chapter 8
//
// Implements a map using an array-based hash table, linear probing collision 
// resolution. 
//
// The remove operation is not supported. Invoking it will result in the 
// unchecked UnsupportedOperationException being thrown.
//
// A map provides (K = key, V = value) pairs, mapping the key onto the value.
// Keys are unique. Keys cannot be null.
//
// Methods throw IllegalArgumentException if passed a null key argument.
//
// Values can be null, so a null value returned by put or get does 
// not necessarily mean that an entry did not exist.
//---------------------------------------------------------------------------
package ch08.maps;

import java.util.Iterator;  

public class HMap<K, V>  implements MapInterface<K,V>
{
  protected MapEntry[] map;
  
  protected final int DEFCAP = 1000;     // default capacity
  protected final double DEFLOAD = 0.75; // default load
  
  protected int origCap;  // original capacity
  protected int currCap;  // current capacity
  protected double load;
  
  protected int numPairs = 0;    // number of pairs in this map
   
  public HMap() 
  {
    map =  new MapEntry[DEFCAP];
    origCap = DEFCAP;
    currCap = DEFCAP;
    load = DEFLOAD;
  }

  public HMap(int initCap, double initLoad) 
  {
    map = new MapEntry[initCap];
    origCap = initCap;
    currCap = initCap;
    load = initLoad;
  }

  private void enlarge()
  // Increments the capacity of the map by an amount 
  // equal to the original capacity.
  {
    // create a snapshot iterator of the map and save current size
    Iterator<MapEntry<K,V>> i = iterator();
    int count = numPairs;

    // create the larger array and reset variables
    map = new MapEntry[currCap + origCap];
    currCap = currCap + origCap; 
    numPairs = 0;
    
    // put the contents of the current map into the larger array
    MapEntry entry;
    for (int n = 1; n <= count; n++)
    {
      entry = i.next();
      this.put((K)entry.getKey(), (V)entry.getValue());
    }
  }
 
  public V put(K k, V v)
  // If an entry in this map with key k already exists then the value 
  // associated with that entry is replaced by value v and the original
  // value is returned; otherwise, adds the (k, v) pair to the map and
  // returns null.
  {
    if (k == null)
      throw new IllegalArgumentException("Maps do not allow null keys.");

    MapEntry<K, V> entry = new MapEntry<K, V>(k, v);
    
    int location = Math.abs(k.hashCode()) % currCap;
    while ((map[location] != null) && !(map[location].getKey().equals(k)))
      location = (location + 1) % currCap;   

    if (map[location] == null)  // k was not in map
    {
      map[location] = entry; 
      numPairs++;
      if ((float)numPairs/currCap > load)
        enlarge();  
      return null;
    }
    else    // k already in map
    {
      V temp = (V)map[location].getValue();
      map[location] = entry; 
      return temp;
    }
  }

  public V get(K k)
  // If an entry in this map with a key k exists then the value associated 
  // with that entry is returned; otherwise null is returned.
  {
    if (k == null)
      throw new IllegalArgumentException("Maps do not allow null keys.");

    int location = Math.abs(k.hashCode()) % currCap;
    while ((map[location] != null) && !(map[location].getKey().equals(k)))
      location = (location + 1) % currCap;   

    if (map[location] == null)  // k was not in map
      return null;
    else                        // k in map
      return (V)map[location].getValue();
  }

  public V remove(K k)
  // Throws UnsupportedOperationException.
  {
    throw new UnsupportedOperationException("HMap does not allow remove.");
  }    

  public boolean contains(K k)
  // Returns true if an entry in this map with key k exists;
  // Returns false otherwise.
  {
    if (k == null)
      throw new IllegalArgumentException("Maps do not allow null keys.");

    int location = Math.abs(k.hashCode()) % currCap;
    while (map[location] != null)
      if (map[location].getKey().equals(k))
        return true;
      else
        location = (location + 1) % currCap;   
    
    // if get this far then no current entry is associated with k
    return false;
  }
   
  public boolean isEmpty()
  // Returns true if this map is empty; otherwise, returns false.
  {
    return (numPairs == 0);
  }
  
  public boolean isFull()
  // Returns true if this map is full; otherwise, returns false.
  {
    return false;  // An HMap is never full
  }

  public int size()
  // Returns the number of entries in this map.
  {
    return numPairs;
  }

  private class MapIterator implements Iterator<MapEntry<K,V>>
  // Provides a snapshot Iterator over this map.
  // Remove is not supported and throws UnsupportedOperationException.
  {
    int listSize = size();
    private MapEntry[] list = new MapEntry[listSize];
    private int previousPos = -1; // previous position returned from list
      
    public MapIterator()
    {
      int next = -1;
      for (int i = 0; i < listSize; i++)
      {
        next++;
        while (map[next] == null)
          next++;
        list[i] = map[next];
      }
    }
      
    public boolean hasNext()
    // Returns true if the iteration has more entries; otherwise returns false.
    {
      return (previousPos < (listSize - 1)) ;
    }
      
    public MapEntry<K,V> next()
    // Returns the next entry in the iteration.
    // Throws NoSuchElementException - if the iteration has no more entries
    { 
      if (!hasNext())
        throw new IndexOutOfBoundsException("illegal invocation of next " + 
                           " in HMap iterator.\n");
      previousPos++;
      return list[previousPos];
    }

    public void remove()
    // Throws UnsupportedOperationException.
    // Not supported. Removal from snapshot iteration is meaningless.
    {
      throw new UnsupportedOperationException("Unsupported remove attempted on " 
                                            + "HMap iterator.\n");
    }
  }

  public Iterator<MapEntry<K,V>> iterator()
  // Returns a snapshot Iterator over this map.
  // Remove is not supported and throws UnsupportedOperationException.

  {
    return new MapIterator();
  }
}
 