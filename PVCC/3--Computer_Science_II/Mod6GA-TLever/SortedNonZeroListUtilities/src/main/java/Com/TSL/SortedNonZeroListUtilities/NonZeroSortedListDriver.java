package Com.TSL.SortedNonZeroListUtilities;


/**
* @author YINGJIN CUI
* version 1.0
* since   2020-03
*
* NonzeroSortedListDriver.java: The driver program for NonzeroList.java
*/

public class NonZeroSortedListDriver{

	
   /**
    * main is the entry point of this program, which creates a new list for sorted non-zero integers, displays
    * information about the list, adds integers to the list, displays information about the list, tries to add more
    * integers to a full list, attempts to remove an integer that is not in the list, removes an integer in the list,
    * displays information about the list, tries to add zero to the list, displays information about the list,
    * removes all data from the list, and displays information about the list.
    * 
    * @param args
    */
	
   public static void main(String[] args){
      
      SortedNonZeroList list=new SortedNonZeroList(5);
      System.out.println("Before adding any data, call list.isEmpty() retrurns: "+list.isEmpty()+ " list.size() returns: "+ list.size());
      list.add(2);
      list.add(12);
      list.add(5);
      list.add(15);
      list.add(9);
      list.print();
      System.out.println("The list is full: "+ list.isFull());
      System.out.println("Try to add more data to the list:");
      list.add(7);
      list.add(200);
      list.remove(89);
      list.remove(5);
      System.out.println("After removing 5.");
      list.print();
      System.out.println("size = "+list.size());
      System.out.println("Try to add 0");
      list.add(0);
      list.print();
      System.out.println("Remove all data from the list");
      list.removeAll();
      list.print();
      
   }
   
}