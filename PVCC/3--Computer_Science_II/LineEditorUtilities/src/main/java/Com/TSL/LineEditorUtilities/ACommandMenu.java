package Com.TSL.LineEditorUtilities;


/**
 * ACommandMenu represents the structure for a command menu of commands sorted in alphabetic order.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class ACommandMenu {

	
	ADoublyLinkedListNode<ACommand> frontNode;
	
	
	/**
	 * ACommandMenu() is the zero-parameter constructor for ACommandMenu, which sets this command menu's reference to
	 * its front node to null.
	 */
	
	public ACommandMenu() {
		
		this.frontNode = null;
		
	}
	
	
    /**
     * contains indicates whether or not this command menu contains a provided command.
     * 
     * @param theCommand
     * @return
     */
    
    private boolean contains(ACommand theCommand) {
    	
    	ADoublyLinkedListNode<ACommand> theCurrentDoublyLinkedListNode = this.frontNode;
    	
    	ACommand theCurrentCommand;
    	while (theCurrentDoublyLinkedListNode != null) {
    		theCurrentCommand = theCurrentDoublyLinkedListNode.providesItsData();
    		if (theCurrentCommand.equals(theCommand)) {
    			return true;
    		}
    		theCurrentDoublyLinkedListNode = theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode();
    	}
    	
    	return false;
    	
    }
	
	
	/**
	 * insert inserts a provided command into this command menu, which is sorted in alphabetic order.
	 * 
	 * @param theCommand
	 * @throws AnInsertsCommandException
	 */
	
    public void inserts(ACommand theCommand) throws AnInsertsCommandException {
    	
    	if (theCommand == null) {
    		throw new AnInsertsCommandException("A command menu found that a reference to a command to insert was null.");
    	}
    	
    	if (contains(theCommand)) {
    		throw new AnInsertsCommandException("A command menu found that it already contained the command to add.");
    	}
    	
    	if (this.frontNode == null) {
    		this.frontNode = new ADoublyLinkedListNode<ACommand>(theCommand, null, null);
    	}
    	
    	else if (theCommand.compareTo(this.frontNode.providesItsData()) < 0) {
			ADoublyLinkedListNode<ACommand> theDoublyLinkedListNodeForTheItem = new ADoublyLinkedListNode<ACommand>(
				theCommand, null, this.frontNode
			);
			this.frontNode.setsItsReferenceToThePreviousNodeTo(theDoublyLinkedListNodeForTheItem);
			this.frontNode = theDoublyLinkedListNodeForTheItem;
    	}
    	
    	else {
    	
	    	ADoublyLinkedListNode<ACommand> theCurrentDoublyLinkedListNode = this.frontNode;
	    	
	    	while ((theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode() != null) &&
	    		   (theCommand.compareTo(theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode().providesItsData()) > 0)) {
	    		theCurrentDoublyLinkedListNode = theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode();
	    	}
	    	
	    	ADoublyLinkedListNode<ACommand> theDoublyLinkedListNodeForTheItem = new ADoublyLinkedListNode<ACommand>(
	    		theCommand, theCurrentDoublyLinkedListNode, theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode()
	    	);
	    	
	    	if (theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode() != null) {
	    		theDoublyLinkedListNodeForTheItem
	    			.providesItsReferenceToTheNextNode()
	    			.setsItsReferenceToThePreviousNodeTo(theDoublyLinkedListNodeForTheItem);
	    	}
	    	
	    	theCurrentDoublyLinkedListNode.setsItsReferenceToTheNextNodeTo(theDoublyLinkedListNodeForTheItem);
    	
    	}
    	
    }
    
    
    /**
     * providesTheFirstInstanceOf provides the first instance of a command in the command menu with the same name and
     * text as a provided command.
     * 
     * @param theCommand
     * @return
     */
    
    public ACommand providesTheFirstInstanceOf(ACommand theCommand) {
    	
    	ADoublyLinkedListNode<ACommand> theCurrentDoublyLinkedListNode = this.frontNode;
    	
    	while (theCurrentDoublyLinkedListNode != null) {
    		if (theCurrentDoublyLinkedListNode.providesItsData().compareTo(theCommand) == 0) {
    			return theCurrentDoublyLinkedListNode.providesItsData();
    		}
    		theCurrentDoublyLinkedListNode = theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode();
    	}
    	
    	return null;
    	
    }
    
    
    /**
     * toString provides a string representation of this command menu.
     */
    
    @Override
    public String toString() {
    	
    	String theRepresentationOfThisCommandMenu = "Command Menu: [\n";
    	
    	ADoublyLinkedListNode<ACommand> theCurrentDoublyLinkedListNode = this.frontNode;
    	while (theCurrentDoublyLinkedListNode != null) {
    		theRepresentationOfThisCommandMenu += "\t" + theCurrentDoublyLinkedListNode.providesItsData() + "\n";
    		theCurrentDoublyLinkedListNode = theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode();
    	}
    	
    	theRepresentationOfThisCommandMenu += "]";
    	
    	return theRepresentationOfThisCommandMenu;
    	
    }
	
}
