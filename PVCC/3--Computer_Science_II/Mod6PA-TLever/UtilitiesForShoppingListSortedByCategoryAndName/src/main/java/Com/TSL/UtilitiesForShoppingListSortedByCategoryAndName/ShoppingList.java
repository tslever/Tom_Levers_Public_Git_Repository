package Com.TSL.UtilitiesForShoppingListSortedByCategoryAndName;

import java.text.NumberFormat;
import java.util.Locale;

/**
 * ShoppingList represents the structure for a shopping list organized by category and then by name.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/24/21
 */

public class ShoppingList 
{
	
	ADoublyLinkedListNode<Item> head;
	int numberOfUniqueItems;
	
	
	/**
	 * ShoppingList() is the zero-parameter constructor for ShoppingList, which sets this list's reference to its first
	 * node to null, and sets this list's number of unique items to 0.
	 */
	
	public ShoppingList() {
		
		this.head = null;
		this.numberOfUniqueItems = 0;
		
	}
	
	
	/**
	 * isEmpty indicates whether or not this list is empty.
	 * @return
	 */
	
    public boolean isEmpty() {
    	return (this.head == null);
    }
    
    
    /**
     * isFull indicates whether or not this list is full.
     * @return
     */
    
    public boolean isFull() {
    	return false;
    }
    
    
    /**
     * size provides this list's number of unique items.
     * @return
     */
    
    public int size() {
    	return this.numberOfUniqueItems;
    }
    
    
    /**
     * totalItems provides the number of total items in this list, considering both unique entries and their quantities.
     * @return
     */
    
    public int totalItems() {
    	
    	int theTotalNumberOfItems = 0;
    	
    	ADoublyLinkedListNode<Item> theCurrentDoublyLinkedListNode = this.head;
    	while (theCurrentDoublyLinkedListNode != null) {
    		theTotalNumberOfItems += theCurrentDoublyLinkedListNode.providesItsData().providesItsQuantity();
    		theCurrentDoublyLinkedListNode = theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode();
    	}
    	
    	return theTotalNumberOfItems;
    	
    }
    
    
    /**
     * grandTotal provides the grand total price of all items on the shopping list.
     * @return
     */
    
    public double grandTotal() {
    	
    	double theTotalOfAllSubtotals = 0.0;
    	
    	ADoublyLinkedListNode<Item> theCurrentDoublyLinkedListNode = this.head;
    	while (theCurrentDoublyLinkedListNode != null) {
    		theTotalOfAllSubtotals += theCurrentDoublyLinkedListNode.providesItsData().Subtotal();
    		theCurrentDoublyLinkedListNode = theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode();
    	}
    	
    	return theTotalOfAllSubtotals;
    	
    }
    
    
    /**
     * printNames outputs the a representation of an array of the names of the unique items in this list.
     */
    
    public void printNames() {
    	
    	String theRepresentationOfTheNamesOfItems = "[";
    	
    	if (this.head != null) {
    		
        	ADoublyLinkedListNode<Item> theCurrentDoublyLinkedListNode = this.head;
    	
	    	while (theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode() != null) {
	    		theRepresentationOfTheNamesOfItems +=
	    			theCurrentDoublyLinkedListNode.providesItsData().providesItsName() + ", ";
	    		theCurrentDoublyLinkedListNode = theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode();
	    	}
	    	
	    	theRepresentationOfTheNamesOfItems +=
	    		theCurrentDoublyLinkedListNode.providesItsData().providesItsName();
    	
    	}
    	
    	theRepresentationOfTheNamesOfItems += "]";
    	
    	System.out.println(theRepresentationOfTheNamesOfItems);
    	
    }
    
    
    /**
     * aStringOfSpacesBasedOn provides a string of spaces based on a string and a width of a column in a table,
     * which will be used to space string values into their appropriate cells.
     *     
     * @param theString
     * @param theWidthOfTheColumn
     * @return
     */
    
    private String aStringOfSpacesBasedOn(String theString, int theWidthOfTheColumn) {
    	
    	String theStringOfSpaces = "";
    	
    	for (int i = 0; i < theWidthOfTheColumn - theString.length(); i++) {
    		theStringOfSpaces += " ";
    	}
    	
    	return theStringOfSpaces;
    	
    }
    
    
    /**
     * aStringRepresentationOf provides a string representation of a United-Stated monetary amount, for the purpose
     * of aligning all subtotals.
     * 
     * @param theUsMonetaryAmount
     * @return
     */
    
    private String aStringRepresentationOf(double theUsMonetaryAmount) {
    	
    	NumberFormat theNumberFormat = NumberFormat.getInstance(Locale.US);
    	theNumberFormat.setMinimumIntegerDigits(4);
    	
    	return theNumberFormat.format(theUsMonetaryAmount);
    	
    }
    
    
    /**
     * print outputs a representation of this shopping list to the standard output stream.
     */
    
    public void print() {
    	
    	System.out.print("Name");
    	int theWidthOfTheNameColumn = 62;
    	System.out.print(aStringOfSpacesBasedOn("Name", theWidthOfTheNameColumn));
    	System.out.print("Quantity");
    	int theWidthOfTheQuantityColumn = 11;
    	System.out.print(aStringOfSpacesBasedOn("Quantity", theWidthOfTheQuantityColumn));
    	System.out.println("Subtotal ($)");
    	System.out.println("=====================================================================================");
    	
    	if (this.head != null) {
    	
	    	String thePresentCategory = "";
	    	int theIndexOfTheItemInThePresentCategory = 1;
	    	
	    	ADoublyLinkedListNode<Item> theCurrentDoublyLinkedListNode = this.head;
	    	while (theCurrentDoublyLinkedListNode != null) {
	    		
	    		Item theItem = theCurrentDoublyLinkedListNode.providesItsData();
	    		if (theItem.providesItsCategory() != thePresentCategory) {
	    			thePresentCategory = theItem.providesItsCategory();
	    			System.out.println("\n" + thePresentCategory + ":");
	    			theIndexOfTheItemInThePresentCategory = 1;
	    		}
	    		
	    		String theIndexedName = theIndexOfTheItemInThePresentCategory + ". " + theItem.providesItsName();
	    		System.out.println(
	    			theIndexedName + aStringOfSpacesBasedOn(theIndexedName, theWidthOfTheNameColumn) +
	    			theItem.providesItsQuantity() + aStringOfSpacesBasedOn(Integer.toString(theItem.providesItsQuantity()), theWidthOfTheQuantityColumn) +
	    			aStringRepresentationOf(theItem.Subtotal())
	    		);
	    		theIndexOfTheItemInThePresentCategory++;
	    		theCurrentDoublyLinkedListNode = theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode();
	    		
	    	}
	    	
    	}
    	
    	System.out.println("=====================================================================================");
    	
    	String theLabeledNumberOfItems = "Number of Items: " + totalItems();
    	System.out.print(
    		theLabeledNumberOfItems +
    		aStringOfSpacesBasedOn(theLabeledNumberOfItems, theWidthOfTheNameColumn + theWidthOfTheQuantityColumn) +
    		"Grand Total: " + aStringRepresentationOf(grandTotal()) + "\n"
    	);
    	
    }
    
    
    /**
     * providesTheFirstInstanceOf provides the first instance of a provided item in the shopping list.
     * 
     * @param theItem
     * @return
     */
    
    public Item providesTheFirstInstanceOf(Item theItem) {
    	
    	ADoublyLinkedListNode<Item> theCurrentDoublyLinkedListNode = this.head;
    	
    	while (theCurrentDoublyLinkedListNode != null) {
    		if (theCurrentDoublyLinkedListNode.providesItsData().compareTo(theItem) == 0) {
    			return theCurrentDoublyLinkedListNode.providesItsData();
    		}
    		theCurrentDoublyLinkedListNode = theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode();
    	}
    	
    	return null;
    	
    }
    
    
    /**
     * insert inserts a provided item into the shopping list.
     * 
     * @param item
     */
    
    public void insert(Item item) {
    	
    	if (item == null) {
    		System.out.println("Warning: A reference to an item to insert into a shopping list was null.");
    		return;
    	}
    	
    	Item theFirstInstanceOfTheItem = providesTheFirstInstanceOf(item);
    	if (theFirstInstanceOfTheItem != null) {
    		theFirstInstanceOfTheItem.increasesItsQuantityBy(item.providesItsQuantity());
    		return;
    	}
    	
    	if (this.head == null) {
    		this.head = new ADoublyLinkedListNode<Item>(item, null, null);
    	}
    	
    	else if (item.compareTo(this.head.providesItsData()) < 0) {
			ADoublyLinkedListNode<Item> theDoublyLinkedListNodeForTheItem = new ADoublyLinkedListNode<Item>(
				item, null, this.head
			);
			this.head.setsItsReferenceToThePreviousNodeTo(theDoublyLinkedListNodeForTheItem);
			this.head = theDoublyLinkedListNodeForTheItem;
    	}
    	
    	else {
    	
	    	ADoublyLinkedListNode<Item> theCurrentDoublyLinkedListNode = this.head;
	    	
	    	while ((theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode() != null) &&
	    		   (item.compareTo(theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode().providesItsData()) > 0)) {
	    		theCurrentDoublyLinkedListNode = theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode();
	    	}
	    	
	    	ADoublyLinkedListNode<Item> theDoublyLinkedListNodeForTheItem = new ADoublyLinkedListNode<Item>(
	    		item, theCurrentDoublyLinkedListNode, theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode()
	    	);
	    	
	    	if (theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode() != null) {
	    		theDoublyLinkedListNodeForTheItem
	    			.providesItsReferenceToTheNextNode()
	    			.setsItsReferenceToThePreviousNodeTo(theDoublyLinkedListNodeForTheItem);
	    	}
	    	
	    	theCurrentDoublyLinkedListNode.setsItsReferenceToTheNextNodeTo(theDoublyLinkedListNodeForTheItem);
    	
    	}
    	
    	this.numberOfUniqueItems++;
    	
    }
    
    
    /**
     * remove removes the first instance of a provided item from the list.
     * 
     * @param item
     */
    
    public void remove(Item item) {
    	
    	if (item == null) {
    		System.out.println("Warning: A reference to an item to remove from a shopping list was null.");
    		return;
    	}
    	
    	Item theFirstInstanceOfTheItem = providesTheFirstInstanceOf(item);
    	if (theFirstInstanceOfTheItem == null) {
    		System.out.println("Warning: The item does not exist in the list.");
    		return;
    	}
    	
    	if (theFirstInstanceOfTheItem == this.head.providesItsData()) {
    		this.head = this.head.providesItsReferenceToTheNextNode();
    	}
    	
    	ADoublyLinkedListNode<Item> theDoublyLinkedListNodeForTheItem = this.head;
    	while (theDoublyLinkedListNodeForTheItem.providesItsData() != theFirstInstanceOfTheItem) {
    		theDoublyLinkedListNodeForTheItem = theDoublyLinkedListNodeForTheItem.providesItsReferenceToTheNextNode();
    	}
    	
    	    	
    	if (theDoublyLinkedListNodeForTheItem.providesItsReferenceToTheNextNode() != null) {
	    	theDoublyLinkedListNodeForTheItem.providesItsReferenceToTheNextNode().setsItsReferenceToThePreviousNodeTo(
	    		theDoublyLinkedListNodeForTheItem.providesItsReferenceToThePreviousNode()
	    	);
    	}
    	
    	theDoublyLinkedListNodeForTheItem.providesItsReferenceToThePreviousNode().setsItsReferenceToTheNextNodeTo(
    		theDoublyLinkedListNodeForTheItem.providesItsReferenceToTheNextNode()
    	);
    	
    }
    
}
