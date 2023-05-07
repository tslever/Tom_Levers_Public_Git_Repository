package Com.TSL.UtilitiesForCalculatingVocabularyDensityAndWordFrequencies;


/**
* @author YINGJIN CUI
* @version 1.0
* since   2020-03
*
* Student name:  Tom Lever
* Completion date: 06/15/21
*
* Note: A link-based implementation of our unsorted Collection ADT
* In this implementation, duplicate elements are allowed
*/

public class LinkedCollectionT<T> implements CollectionInterface<T>{

	
   //You are NOT allowed to add other instance variables for this class!

   private LLNode<T> head;
   
   
   
   /**
    * LinkedCollectionT is the zero-parameter constructor for LinkedCollectionT, which sets this linked collection's
    * head to the null reference.
    */
   
   public LinkedCollectionT() {
	   
      this.head = null;
      
   }
   
   
   public void add(T ele){

      // *** Student task #1 ***  

      /* 
      Requirements: 
      Theoretically, a linked list node implementation has no capacy limit.
      Because it is unsorted, you may simply append a new element to the end of the collection
 
       *** Enter your code below *** 
       */

      LLNode<T> theLinkedListNodeForTheElement = new LLNode<T>(ele);
      
      if (this.head == null) {
    	  this.head = theLinkedListNodeForTheElement;
      }
      else {
	      LLNode<T> theCurrentLinkedListNode = this.head;
	      
	      while (theCurrentLinkedListNode.getNext() != null) {
	    	  theCurrentLinkedListNode = theCurrentLinkedListNode.getNext();
	      }
	      
	      theCurrentLinkedListNode.setNext(theLinkedListNodeForTheElement);
      }
	   
   }

   
    public int remove(T ele){

      // *** Student task #2 ***  

      /* 
      Requirements: 
      Removes all elements from this collection such that each of them equals to ele [equals(ele) returns true]
      and returns the number of elements being removed.

       *** Enter your code below *** 
      */
    	
    	int theNumberOfElementsRemoved = 0;
    	
    	while ((this.head != null) && (this.head.getData().equals(ele))) {
    		this.head = this.head.getNext();
    		theNumberOfElementsRemoved++;
    	}
    	
    	if (this.head == null) {
    		return theNumberOfElementsRemoved;
    	}
    	
    	LLNode<T> thePreviousLinkedListNode = this.head;
    	LLNode<T> theCurrentLinkedListNode = this.head.getNext();    	
    	
    	while (theCurrentLinkedListNode != null) {
    		
    		if (theCurrentLinkedListNode.getData().equals(ele)) {
    			thePreviousLinkedListNode.setNext(theCurrentLinkedListNode.getNext());
    			theNumberOfElementsRemoved++;
    		}
    		else {
    			thePreviousLinkedListNode = thePreviousLinkedListNode.getNext();
    		}
    		
			theCurrentLinkedListNode = theCurrentLinkedListNode.getNext();
    		
    	}
    	
    	return theNumberOfElementsRemoved;

   }
   
    
   public boolean isFull() {

      // *** Student task #3 ***  

      /* 
      Note: Theoretically, a linked list node implementation has no capacy limit

       *** Enter your code below *** 
     */
	   
	   return false;
 
   }
   
   
   public boolean isEmpty() {

      // *** Student task #4 ***  

      /* 
      Requirements: 
      To check if the collection list is empty

       *** Enter your code below *** 
     */
	   
	   return (this.head == null);


   }
   
   
   public int size() {
 
      // *** Student task #5 ***  

      /* 
      Requirements: 
      To return the number of items in the collection list

       *** Enter your code below *** 
     */
	   
	   int theNumberOfElements = 0;
	   
	   LLNode<T> theCurrentLinkedListNode = this.head;
	   while (theCurrentLinkedListNode != null) {
		   theNumberOfElements++;
		   theCurrentLinkedListNode = theCurrentLinkedListNode.getNext();
	   }
	   
	   return theNumberOfElements;

   }
   
   
   public T find(T target) {

     // *** Student task #6 ***  

      /* 
      Requirements: 
      To find and return the first occurance of the target object (compared with equals method), if not found, return null

       *** Enter your code below *** 
      */
	   
      LLNode<T> theLinkedListNodePotentiallyMatchingTheTarget = this.head;
      
      while (theLinkedListNodePotentiallyMatchingTheTarget != null) {
    	
    	  if (theLinkedListNodePotentiallyMatchingTheTarget.getData().equals(target)) {
    		  return theLinkedListNodePotentiallyMatchingTheTarget.getData();
    	  }
    	  
    	  theLinkedListNodePotentiallyMatchingTheTarget = theLinkedListNodePotentiallyMatchingTheTarget.getNext();
    	  
      }
      
      return null;

   }
      
   public void sort() {
      //Use quick sort to sort the lememnts
      //get all elements in the collection and put them into an array,
      //then call quick sort to sort the array. After sorintg restore the collection
      T[] arr=(T[])new Object[size()];
      LLNode<T> tmp=head;
      for(int i=0; i<arr.length; i++){
         arr[i]=tmp.getData();
         tmp=tmp.getNext();
      }
      new QuickSort<T>().quickSort(arr);
      //restore the collection;
      head=null;
      for(int i=0; i<arr.length; i++){
         add(arr[i]);
      }
   }

   
   /**
    * print displays a representation of this collection, including the bases for words along with their frequencies.
    */
   
   public void print() {
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