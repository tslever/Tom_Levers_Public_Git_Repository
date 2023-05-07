package Com.TSL.CollectionAbstractDataTypeUtilities;


/**
 * CollectionTTest encapsulates a method that tests all of the methods of the CollectionT class.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/14/21
 */

public class CollectionTTest
{
	
	/**
	 * main tests all of the methods of the CollectionT class.
	 * 
	 * @param args
	 */
	
    public static void main( String[] args )
    {
        CollectionT<Integer> theCollectionOfIntegers = new CollectionT<Integer>(10);
        
        System.out.println("Before adding any item, the collection is empty: " + theCollectionOfIntegers.isEmpty());
        System.out.println("Before adding any item, the collection is:");
        theCollectionOfIntegers.print();
        System.out.println();
        
        System.out.println("After adding four items, the collection is:");
        for (int i = 0; i < 3; i++) {
        	theCollectionOfIntegers.add(i + 1);
        }
        theCollectionOfIntegers.add(1);
        theCollectionOfIntegers.print();
        System.out.println("The size of the collection is: " + theCollectionOfIntegers.size() + "\n");
        
        System.out.println("After removing the integer at index 1:");
        theCollectionOfIntegers.remove(1);
        theCollectionOfIntegers.print();
        System.out.println();
        
        int theNumberOfIntegersRemoved = theCollectionOfIntegers.remove((Integer) 1);
        System.out.println("After removing " + theNumberOfIntegersRemoved + " integers equal to 1:");
        theCollectionOfIntegers.print();
        System.out.println();
        
        // At this point, for theCollectionOfIntegers, the following have been tested:
        // - CollectionT(int size)
        // - add(T ele)
        // - remove(int index)
        // - remove(T ele)
        
        System.out.println("After adding 1 at index 0 and displacing integers to the right:");
        theCollectionOfIntegers.add(1, 0);
        theCollectionOfIntegers.print();
        System.out.println();
        
        System.out.println("After adding 2 at index 1 and displacing integers to the right:");
        theCollectionOfIntegers.add(2, 1);
        theCollectionOfIntegers.print();
        System.out.println();
        
        System.out.println("After adding seven more integers:");
        for (int i = 4; i <= 10; i++) {
        	theCollectionOfIntegers.add(i);
        }
        theCollectionOfIntegers.print();
        System.out.println("The size and capacity of the collection:");
        System.out.println(theCollectionOfIntegers.size() + ", " + theCollectionOfIntegers.providesItsCapacity());
        System.out.println("After adding one more integer:");
        theCollectionOfIntegers.add(11);
        theCollectionOfIntegers.print();
        System.out.println("The size and capacity of the collection:");
        System.out.println(theCollectionOfIntegers.size() + ", " + theCollectionOfIntegers.providesItsCapacity() + "\n");
        
        System.out.println("After providing the integer in the collection at index 9:");
        System.out.println(theCollectionOfIntegers.get(9) + "\n");
        
        System.out.println("After providing an integer equal to 9:");
        System.out.println(theCollectionOfIntegers.get((Integer) 9) + "\n");
        
        System.out.println("After testing to see whether this collection contains 5:");
        System.out.println(theCollectionOfIntegers.contains(5));
        
    }
    
}
