package Com.TSL.SortedDoublyLinkedListBasedCollectionUtilities;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        ASortedDoublyLinkedListBasedCollection<String> theSortedCollectionOfStrings =
        	new ASortedDoublyLinkedListBasedCollection<String>();
        
        theSortedCollectionOfStrings.add("1");
        theSortedCollectionOfStrings.add("9");
        theSortedCollectionOfStrings.add("7");
        theSortedCollectionOfStrings.add("8");
        theSortedCollectionOfStrings.add("6");
        theSortedCollectionOfStrings.add("4");
        theSortedCollectionOfStrings.add("5");
        theSortedCollectionOfStrings.add("5");
        theSortedCollectionOfStrings.add("3");
        theSortedCollectionOfStrings.add("1");
        theSortedCollectionOfStrings.add("2");
        theSortedCollectionOfStrings.add("9");
        
        System.out.println("The sorted collection after adding somewhat randomly strings representing integers:");
        System.out.println(theSortedCollectionOfStrings);
        System.out.println();
        
        theSortedCollectionOfStrings.remove("5");
        
        System.out.println("The sorted collection after removing a string representing an integer:");
        System.out.println(theSortedCollectionOfStrings);
    }
}
