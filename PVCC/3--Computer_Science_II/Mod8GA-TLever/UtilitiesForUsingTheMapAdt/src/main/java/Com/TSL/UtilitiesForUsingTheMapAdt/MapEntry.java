package Com.TSL.UtilitiesForUsingTheMapAdt;


//---------------------------------------------------------------------------
//MapEntry.java              by Dale/Joyce/Weems                   Chapter 8
//
//Provides key, value pairs for use with a Map.
//Keys are immutable.
//---------------------------------------------------------------------------

public class MapEntry<K, V>
{
	protected K key;
	protected V value;
	
	
	/**
	 * MapEntry(K k, V v) is the two-parameter constructor for MapEntry, which sets the key and value of this map entry
	 * to provided keys and values.
	 * @param k
	 * @param v
	 */
	
	MapEntry(K k, V v)
	{
	 key = k; value = v;
	}
	
	
	/**
	 * getKey provides this map entry's key.
	 * @return
	 */
	
	public K getKey()  {return key;}
	
	
	/**
	 * getValue provides this map entry's value.
	 */
	
	public V getValue(){return value;}
	
	
	/**
	 * setValue sets this map entry's value.
	 * @param v
	 */
	
	public void setValue(V v){value = v;}
	
	
	@Override
	public String toString()
	// Returns a string representing this MapEntry.
	{
	 return "Key  : " + key + "\nValue: " + value;
	}
	
}