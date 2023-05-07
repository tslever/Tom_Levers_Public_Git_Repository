package Com.TSL.NonZeroListUtilities;


/**
* @author YINGJIN CUI
* version 1.0
* since   2020-03 
*
* Student name:  Tom Lever
* Completion date: 06/20/21
*
* NonzeroList represents the structure of an array-based list of non-zero integers.
*/

public class NonzeroList{
	

   private int[] data;
   private int index;   //the location where new data will be added to the array.
   private int numberOfElements;
   
   
   /**
    * NonzeroList(int cap) is the one-parameter constructor for NonzeroList, which initializes this list's array of
    * integers to a new array with the provided capacity, initializes this list's location for adding new integers to
    * zero, and initializes this list's number of integers to 0. 
    * 
    * @param cap
    */
   
   public NonzeroList(int cap) {
	   
      data = new int[cap];
      index=numberOfElements=0;
      
   }
   
   
   public void add(int num){

      	// *** Student task #1 *** 

      	/*
	Requirements:
        -if num is zero, print message saying that zero is not allowed in a nonzerolist
        -if it is full, display message "The NonzeroList is full."
        - Else, add num to this list's array of integers, and increment index and numberOfElements.

        *** Enter your code below *** 
	*/

	   if (num == 0) {
		   System.out.println("zero is not allowed in a nonzerolist.");
		   return;
	   }
	   
	   if (isFull()) {
		   System.out.println("The NonzeroList is full.");
		   return;
	   }
	   
	   this.data[this.index] = num;
	   this.index++;
	   this.numberOfElements++;

   }
   
   public void removeData(int target) {

      // *** Student task #2 ***  

      	/* 
	Requirements: 
      	-Remove the first occurrence of the target value in the NonzeroList
      	-Move the last item in the NonzeroList to the above position. 
	-You may shift after the removed item but it's not sufficient. The big-O of Moving the last item to the
      	  removed target location if O(1), while complexity of shifting algorithm is O(N)-
      	  Can you figure out why?
      	-If the target value does not exist, print message: "Target value does not exist."

       	*** Enter your code below *** 
     	*/
	   
	   for (int i = 0; i < this.numberOfElements; i++) {
		   if (this.data[i] == target) {
			   this.data[i] = this.data[this.numberOfElements - 1];
			   this.data[this.numberOfElements - 1] = 0;
			   this.index--;
			   this.numberOfElements--;
			   return;
		   }
	   }
	   
	   System.out.println("Target value " + target + " does not exist.");

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
    * size provides the number of elements in this list.
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