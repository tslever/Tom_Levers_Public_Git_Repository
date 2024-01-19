package Com.TSL.UtilitiesForCalculatingVocabularyDensityAndWordFrequencies;


/**
* @author YINGJIN CUI
* version 1.0
* since   2020-03-30
*
* CollectionInterface.java: interface file
*
* Generic type is used to allow you to write a general, generic methods that works 
* with different types-allowing for code re-use.
* A collection allows addition, removal, and access of elements.
* In this implemention, null is not allowed, duplicate elements may or may not be permitted.
*/

public interface CollectionInterface<T>{
   
   void add(T ele);
   //Add an element to the collection
    
   T find(T ele);
   // Returns the first element e inn the collection such that
   // e.equals(ele) returns true; if no such element found, return null
  
   int remove(T ele);
   // Removes all elements from this collection such that each of them equals to ele [equals(ele) returns true]
   // and returns the number of elements being removed.
    
   boolean isFull();
   // Returns true if this collection is full; otherwise, returns false.
   
   boolean isEmpty();
   // Returns true if this collection is empty; otherwise, returns false.
   
   int size();
   // Returns the number of elements in this collection.
   
   void sort();
   // sorts the words in this collection in descending order by frequency.
   
   void print();
   // Print all elements
}