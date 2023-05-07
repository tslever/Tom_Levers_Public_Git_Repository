package Com.TSL.SortedLinkBasedCollectionUtilities;


/**
* @author YINGJIN CUI
* @version 1.0
* since   2020-03
*
* Student name:  Tom Lever
* Completion date: 06/16/21
*
* SortedCollectionT.txt: the template file of SortedCollectionT.java
* Student tasks: implement tasks #1 and #2 as specified in this file
*
* A link-based implementation of our sorted Collection ADT
* In this implementation, duplicate elements are not allowed
*/

public class SortedCollectionT<T extends Comparable<T>> implements SortedCollectionInterface<T>{

	
   private LLNode<T> head;
   
   
   
   /**
    * SortedCollectionT() is the zero-parameter constructor for SortedCollectionT, which sets this sorted collection's
    * head reference to null.
    */
   
   public SortedCollectionT(){
	   
      head=null;
      
   }
   
   
   public void add(T ele){

      // *** Student task #1 ***  

      /* 
      Requirements: 
      If the collection is full then double the array size before adding
      However, theoretically, a linked list node implementation has no capacy limit.
      In this implementation, duplicate elements are not allowed

       *** Enter your code below *** 
       */
   
	   
	   // When duplicates are not allowed
	   
	   if (find(ele) != null) {
		   throw new AnAddingDuplicateElementException("Adding a duplicate element was requested.");
	   }
	   
	   
	   // no duplication is allowed, so just <, not <=
	   
	   if (this.head == null || ele.compareTo(this.head.getData()) < 0) {
		   this.head = new LLNode<T>(ele, this.head);
	   }
	   
	   else {
		   
		   LLNode<T> theCurrentLinkedListNode = this.head;
		   
		   while ((theCurrentLinkedListNode.getNext() != null) &&
				  (ele.compareTo(theCurrentLinkedListNode.getNext().getData()) > 0)) {
			   theCurrentLinkedListNode = theCurrentLinkedListNode.getNext();
		   }

		 // if the value is the same, do not add to the list, b/c duplication is not allowed
		   
         if (!ele.equals(theCurrentLinkedListNode.getData())) {

            LLNode<T> theLinkedListNodeForTheElement = new LLNode<T>(ele, theCurrentLinkedListNode.getNext());
            theCurrentLinkedListNode.setNext(theLinkedListNodeForTheElement);

         }
		   
	   }

   }
   
   
   public void remove(T ele){
      // *** Student task #2 ***  

      /* 
      Requirements: 
      Removes the element e from this collection such that e.equals(ele)

       *** Enter your code below *** 
      */
	  
	  if (isEmpty()) {
		  throw new ACollectionUnderflowException("Exception: remove for an empty collection requested.");
	  }
	  
	  if (find(ele) == null) {
		  return;
	  }
	  //you may call find method to determine if the ele exists or not before removing
          //**** it is OK w/o using the method, but from software engineering pespective, it is a good practice to use given method

	  // For removing one element equal to the provided element from the collection
	  if (ele.equals(this.head.getData())) {
		  this.head = this.head.getNext();
		  return;
	  }
	  
	  // For removing all elements equal to the provided element from the collection
	  //while ((this.head != null) && (ele.equals(this.head.getData()))) {
		//  this.head = this.head.getNext(); 
        //          //**** need to return from here after removing the first ele on the list 
	  //}
	  
	  //if (this.head == null) {
		//  return;
	  //}
	  
	  LLNode<T> thePreviousLinkedListNode = this.head;
	  LLNode<T> theCurrentLinkedListNode = this.head.getNext();
	  
	  while (theCurrentLinkedListNode != null) {
		  if (ele.equals(theCurrentLinkedListNode.getData())) {
			  thePreviousLinkedListNode.setNext(theCurrentLinkedListNode.getNext());//**** need break;
			  break; // For removing one element equal to the provided element from the collection
		  }
		  else {
			  thePreviousLinkedListNode = theCurrentLinkedListNode;
			  // For removing one element equal to the provided element from the collection
			  theCurrentLinkedListNode = theCurrentLinkedListNode.getNext();
		  }
		  // For removing all elements equal to the provided element from the collection
		  //theCurrentLinkedListNode = theCurrentLinkedListNode.getNext(); //**** it is OK, it would be more readable if it is placed inside else
                                                                                 //
	  }

   }
   
   
   /**
    * isFull indicates whether or not this sorted collection is full.
    */
   
   public boolean isFull(){//theoretically, a linked list node implementation has no capacy limit
      return false;
   }
   
   
   /**
    * isEmpty indicates whether or not this sorted collection is empty.
    */
   
   public boolean isEmpty(){
      return head==null;
   }
   
   
   /**
    * size provides the number of elements in this sorted collection.
    */
   
   public int size(){
      LLNode<T> tmp=head;
      int count=0;
      while(tmp !=null){
         tmp=tmp.getNext();
         count++;
      }
      return count;
   }
   
   
   /**
    * find provides an element in the sorted collection that is equal to an input element, or provides a null reference.
    */
   
   public T find(T ele){
      LLNode<T> tmp=head;
      while(tmp != null){
         if(tmp.getData().equals(ele))
            return tmp.getData();
         tmp = tmp.getNext();
      }
      return null;
   }
   
   
   /**
    * print displays a representation of this sorted collection.
    */
   
   public void print(){
      String res="";
      LLNode<T> tmp=head;
      while(tmp!=null){
         res += tmp.getData().toString()+", ";
         tmp = tmp.getNext();
      }
      if(res.length()>0){
         res = res.substring(0, res.length() -2);
      }
      System.out.println("["+res+"]");
   }
   
}