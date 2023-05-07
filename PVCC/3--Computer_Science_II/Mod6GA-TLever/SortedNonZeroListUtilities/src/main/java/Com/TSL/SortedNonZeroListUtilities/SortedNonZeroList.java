package Com.TSL.SortedNonZeroListUtilities;


/**
* @author YINGJIN CUI
* version 1.0
* since   2020-03-23 15:01
*
* Student name:  Tom Lever
* Completion date: 06/21/21
*
* SortedNonZeroList represents the structure for a list of non-zero integers.
*/

public class SortedNonZeroList {
	
   private int[] data;
   private int index; //the location where new data will be added to the array.
   private int numberOfElements;
   
   
   /**
    * SortedNonZeroList(int cap) is the one-parameter constructor for SortedNonZeroList, which sets this list's
    * array of integers to a new array of integers with a capacity of cap, sets this list's index to 0, and sets this
    * list's numberOfElements to 0. 
    * 
    * @param cap
    */
   
   public SortedNonZeroList(int cap) {
	   
      data = new int[cap];
      index=numberOfElements=0;
      
   }
   
   
   public void add(int num){

      // *** Student task #1 ***  

      /* 
	Requirements: 
	-if num is zero, print message saying that zero is not allowed in a nonzerolist
      	-if it is full, display message "The NonzeroList is full."
      	-if the number is duplicate, display message "This number is in the list. A duplicated number can't be added to the list."
      	-num is inserted into the list so that the list is sorted in ascending order

       *** Enter your code below *** 
      */
	   
	   if (num == 0) {
		   System.out.println("Zero is not allowed in a nonzerolist.");
		   return;
	   }
	   
	   if (isFull()) {
		   System.out.println("The NonzeroList is full.");
		   return;
	   }
	   
	   if (indexOf(num) > -1) {
		   System.out.println("This number is in the list. A duplicate number can't be added to the list.");
		   return;
	   }
	   
	   int theIndexAtWhichToInsertTheProvidedInteger = 0;
	   while ((theIndexAtWhichToInsertTheProvidedInteger < this.index) &&
			  (num > this.data[theIndexAtWhichToInsertTheProvidedInteger])) {
		   theIndexAtWhichToInsertTheProvidedInteger++;
	   }
	   
	   for (int i = this.index; i > theIndexAtWhichToInsertTheProvidedInteger; i--) {
		   this.data[i] = this.data[i - 1];
	   }
	   
	   this.data[theIndexAtWhichToInsertTheProvidedInteger] = num;
	   
	   this.index++;
	   this.numberOfElements++;

   }
   
   public void remove(int target){

     // *** Student task #2 ***  

      /* 
	Requirements:
      	-If the target value does not exist, print message: "Target value does not exist."
      	-Remove the target value in the NonzeroList if the value exists
      	-Shift all items after the removed one so the order is maintained.

       *** Enter your code below *** 
     */
	   
	   if (indexOf(target) == -1) {
		   System.out.println("Target value does not exist.");
		   return;
	   }
	   
	   for (int i = indexOf(target); i < this.size() - 1; i++) {
		   this.data[i] = this.data[i + 1];
	   }
	   
	   this.data[this.index - 1] = 0;
	   this.index--;
	   this.numberOfElements--;

   }
   
    public void removeAll(){

     // *** Student task #3 ***  

      /* 
	Requirements:
        -remove all data from the list. The list will be empty 

        *** Enter your code below *** 
     */
    	
    	for (int i = 0; i < size(); i++) {
    		this.data[i] = 0;
    	}
    	
    	this.index = 0;
    	this.numberOfElements = 0;

    }

    
   public int indexOf(int target){ // return the index of the first occurrence of target in the data array
      for(int i=0; i<numberOfElements; i++){
         if(data[i]==target)
            return i;
      }
      return -1; // not found
   }

   
   /**
    * isFull indicates whether or not this list is full.
    * @return
    */
   
   public boolean isFull(){
      return numberOfElements==data.length;
   }

   
   /**
    * isEmpty indicates whether or not this list is empty.
    * @return
    */
   
   public boolean isEmpty(){
      return numberOfElements==0;
   }

   
   /**
    * size provides this list's number of elements.
    * @return
    */
   
   public int size(){
      return numberOfElements;
   }
   
   
   /**
    * print outputs a representation of this list to the standard output stream.
    */

   public void print(){
      System.out.print("[");
      for(int i=0; i<numberOfElements; i++){
         System.out.print(data[i]);
         if(i<numberOfElements-1)
            System.out.print(", ");
      }
      System.out.println("]");
   }
}