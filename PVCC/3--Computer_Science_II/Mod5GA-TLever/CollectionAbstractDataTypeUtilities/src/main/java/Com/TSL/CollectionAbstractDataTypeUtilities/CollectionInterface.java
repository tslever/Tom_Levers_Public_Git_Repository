package Com.TSL.CollectionAbstractDataTypeUtilities;


/**
* @author YINGJIN CUI
* version 1.0
* since   2020-03-30 
*
* CollectionInterface.java - 
* Note: Generic type is used to allow you to write a general, generic methods that works 
* with different types-allowing for code re-use.
* A collection allows addition, removal, and access of elements.
* In this implemention, null is not allowed, but duplicate elements are permitted.
*/

public interface CollectionInterface<T>{
   
   void add(T ele);
   //Add an element to the collection
   
   void add(T ele, int index);
   //Add an element in the index location
   //if index>number of element of the collection, then append the ele to the collection
   
   T get(T ele);
   // Returns an element e from this collection such that e.equals(ele).
   // If no such element exists, returns null.
  
   T get(int index);
   // Returns the (index+1)th element in the collection.
   // If index<0 or index>= the number of elements in this collection, returns null.
   
   boolean contains(T ele);
   // Returns true if this collection contains an element e such that
   // e.equals(ele) returns true; otherwise returns false.
  
   int remove(T ele);
   // Removes all elements from this collection such that each of them equals to ele [equals(ele) returns true]
   // and returns the number of elements being removed.
   
   boolean remove(int index);
   // Removes the element at index position, if index<size() returns true after removing, otherwise, returns false

   int indexOf(T ele);
   // returns index of the first element e such that e.equals(ele) returns true
   // If no such element exists, returns -1
   
   boolean isFull();
   // Returns true if this collection is full; otherwise, returns false.
   
   boolean isEmpty();
   // Returns true if this collection is empty; otherwise, returns false.
   
   int size();
   // Returns the number of elements in this collection.
   
   void print();
   // Print all elements
   
}