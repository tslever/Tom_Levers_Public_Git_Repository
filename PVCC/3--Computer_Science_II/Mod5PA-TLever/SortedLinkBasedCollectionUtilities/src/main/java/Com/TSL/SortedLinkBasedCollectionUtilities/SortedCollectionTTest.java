package Com.TSL.SortedLinkBasedCollectionUtilities;


import org.junit.jupiter.api.Test;


/**
 * SortedCollectionTTest encapsulates a JUnit test of add and remove methods in the SortedCollectionT class.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/16/21
 */

public class SortedCollectionTTest {

	
	/**
	 * testSortedCollectionT tests add and remove methods in the SortedCollectionT class by adding to a sorted
	 * collection-- somewhat randomly --strings representing integers, and removing a string.
	 */
	
	@Test
	public void testSortedCollectionT() {
		
		System.out.println("Running testSortedCollectionT.\n");
		
        SortedCollectionT<String> theSortedCollectionOfStrings = new SortedCollectionT<String>();
        
        theSortedCollectionOfStrings.add("1");
        //theSortedCollectionOfStrings.add("9");
        theSortedCollectionOfStrings.add("7");
        theSortedCollectionOfStrings.add("8");
        theSortedCollectionOfStrings.add("6");
        theSortedCollectionOfStrings.add("4");
        //theSortedCollectionOfStrings.add("5");
        theSortedCollectionOfStrings.add("5");
        theSortedCollectionOfStrings.add("3");
        //theSortedCollectionOfStrings.add("1");
        theSortedCollectionOfStrings.add("2");
        theSortedCollectionOfStrings.add("9");
        
        System.out.println("The sorted collection after adding somewhat randomly strings representing integers:");
        theSortedCollectionOfStrings.print();
        System.out.println();
        
        theSortedCollectionOfStrings.remove("9");
        
        System.out.println("The sorted collection after removing a string representing an integer:");
        theSortedCollectionOfStrings.print();
        System.out.println();
		
	}
	
	
}
