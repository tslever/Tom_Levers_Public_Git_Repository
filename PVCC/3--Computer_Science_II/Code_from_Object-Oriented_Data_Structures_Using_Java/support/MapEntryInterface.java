//---------------------------------------------------------------------------
// MapEntryInterface.java        by Dale/Joyce/Weems                Chapter 8
//
// Provides key, value pairs for use with a Map.
//---------------------------------------------------------------------------
package support;

public interface MapEntryInterface<K, V>
{
  K getKey();
  // Returns the key of this MapEntry.
  
  K getValue();
  // Returns the value of this MapEntry.

  String toString();
  // Returns a string representing this MapEntry.
}
 