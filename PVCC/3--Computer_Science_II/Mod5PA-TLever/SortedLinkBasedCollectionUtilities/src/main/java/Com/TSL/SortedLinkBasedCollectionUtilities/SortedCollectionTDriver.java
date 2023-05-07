package Com.TSL.SortedLinkBasedCollectionUtilities;


import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Scanner;


/**
 * SortedCollectionTDriver encapsulates the entry point of this program, which scans a text file for strings delimited
 * by whitespace; reads and demotes to lower case those strings; adds the strings to a sorted linked list based
 * collection, displays the sorted collection, and displays the collection after removing two strings in the
 * collection.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/16/21
 */

public class SortedCollectionTDriver 
{
	
	/**
	 * main is the entry point of this program, which scans a text file for strings delimited by whitespace; reads and
	 * demotes to lower case those strings; adds the strings to a sorted linked list based collection, displays the
	 * sorted collection, and displays the collection after removing two strings in the collection.
	 * 
	 * @param args
	 * @throws FileNotFoundException
	 */
	
    public static void main(String[] args) throws FileNotFoundException
    {
    	
        String theFilename = args[0];
        FileReader theFileReader = new FileReader(theFilename);
        
        Scanner theScannerOfTheTextFile = new Scanner(theFileReader);
        
        SortedCollectionT<String> theSortedCollectionOfStrings = new SortedCollectionT<String>();
        
        while (theScannerOfTheTextFile.hasNext()) {
        	
        	String theStringToPotentiallyAdd = theScannerOfTheTextFile.next().toLowerCase();
        	if (theSortedCollectionOfStrings.find(theStringToPotentiallyAdd) == null) {
        		theSortedCollectionOfStrings.add(theStringToPotentiallyAdd);
        	}
        	
        }
        
        theScannerOfTheTextFile.close();
        
        System.out.println("The sorted collection of strings after parsing " + theFilename + ":");
        theSortedCollectionOfStrings.print();
        System.out.println();
        
        System.out.println("The sorted collection of strings after removing \"k\" and \"h\":");
        theSortedCollectionOfStrings.remove("k");
        theSortedCollectionOfStrings.remove("h");
        theSortedCollectionOfStrings.print();
        
    }
    
}
