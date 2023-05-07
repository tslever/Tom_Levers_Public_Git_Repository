package Com.TSL.SortedLinkBasedCollectionUtilities;


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

public interface SortedCollectionInterface<T>{
   
   void add(T ele);
   //Add an element to the collection
    
   T find(T ele);
   // Returns the first element e inn the collection such that
   // e.equals(ele) returns true; if no such element found, return null
  
   void remove(T ele);
   // Removes the element e from this collection such that e.equals(ele)
    
   boolean isFull();
   // Returns true if this collection is full; otherwise, returns false.
   
   boolean isEmpty();
   // Returns true if this collection is empty; otherwise, returns false.
   
   int size();
   // Returns the number of elements in this collection.
   
   void print();
   // Print all elements
}