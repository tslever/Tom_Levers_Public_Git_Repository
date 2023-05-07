package Com.TSL.DoublyLinkedListUtilities;


/**
 * DoublyLinkedListDriver encapsulates the entry point of this program, which tests creation and use of a doubly linked
 * list.
 *
 */

public class DoublyLinkedListDriver 
{
	
	
	/**
	 * main encapsulates the entry point of this program, which tests creation and use of a doubly linked list.
	 * 
	 * @param args
	 */
	
    public static void main( String[] args )
    {
    	System.out.println("The doubly linked list is,\n");
    	
        ADoublyLinkedList<String> theDoublyLinkedList = new ADoublyLinkedList<String>();
        System.out.println("after creating a new doubly linked list,\n" + theDoublyLinkedList + "\n");
        
        theDoublyLinkedList.addsToItsHead("A");
        System.out.println("after adding \"A\" to its head,\n" + theDoublyLinkedList + "\n");            
        
        theDoublyLinkedList.addsToItsHead("B");
        System.out.println("after adding \"B\" to its head,\n" + theDoublyLinkedList + "\n");
        
        theDoublyLinkedList.addsToItsTail("C");
        System.out.println("after adding \"C\" to its tail,\n" + theDoublyLinkedList + "\n");
        
        theDoublyLinkedList.addsToItsTail("D");
        System.out.println("after adding \"D\" to its tail,\n" + theDoublyLinkedList + "\n");
        
        theDoublyLinkedList.insertsAtItsMidpoint("E");
        System.out.println("after inserting \"E\" at its midpoint,\n" + theDoublyLinkedList + "\n");
        
        theDoublyLinkedList.removesTheNodeAtItsHead();
        System.out.println("after removing the node at its head,\n" + theDoublyLinkedList + "\n");
        
        theDoublyLinkedList.addsToItsTail("B");
        System.out.println("after adding \"B\" to its tail,\n" + theDoublyLinkedList + "\n");
        
        theDoublyLinkedList.addsToItsTail("F");
        System.out.println("after adding \"F\" to its tail,\n" + theDoublyLinkedList + "\n");
        
        theDoublyLinkedList.setsItsCurrentNodeToItsTail();
        System.out.println(
        	"after setting its current node to its tail, an object with a current node with a string " +
        	theDoublyLinkedList.providesTheElementOfItsCurrentNode() + ".\n"
        );
        
        theDoublyLinkedList.movesItsCurrentNodeAwayFromItsTailAndTowardsItsHead();
        System.out.println(
        	"after moving its current node away from its tail and towards it head, an object with a current node with " +
        	"a string " + theDoublyLinkedList.providesTheElementOfItsCurrentNode() + ".\n"
        );
        
        theDoublyLinkedList.removesTheNodeAtItsMidpoint();
        System.out.println("after removing the node at its midpoint,\n" + theDoublyLinkedList + "\n");
        
        theDoublyLinkedList.addsToItsHead("D");
        System.out.println("after adding \"D\" to its head,\n" + theDoublyLinkedList + "\n");
        
        if (theDoublyLinkedList.finds("B") != null) {
        	System.out.println("an object that contains a node with an element \"B\".\n");
        }
        else {
        	System.out.println("an object that does not contain a node with an element \"B\".\n");
        }
        
        if (theDoublyLinkedList.finds("G") != null) {
        	System.out.println("an object that contains a node with an element \"G\".\n");
        }
        else {
        	System.out.println("an object that does not contain a node with an element \"G\".\n");
        }
        
        System.out.println("sleeps as\n" + theDoublyLinkedList + "\n");
        
    }
}
