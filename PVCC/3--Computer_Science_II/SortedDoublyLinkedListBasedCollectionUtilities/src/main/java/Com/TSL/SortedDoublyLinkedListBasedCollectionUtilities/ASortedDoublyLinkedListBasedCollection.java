package Com.TSL.SortedDoublyLinkedListBasedCollectionUtilities;


public class ASortedDoublyLinkedListBasedCollection<T extends Comparable<T>> {

	
	ADoublyLinkedListNode<T> head;
	
	
	public ASortedDoublyLinkedListBasedCollection() {
		this.head = null;
	}
	
	
	public void add(T theElementToAdd) {

		if (this.head == null) {
			this.head = new ADoublyLinkedListNode<T>(theElementToAdd, null, null);
		}
		
		else if (theElementToAdd.compareTo(this.head.providesItsData()) <= 0) {
			ADoublyLinkedListNode<T> theDoublyLinkedListNodeForTheElement = new ADoublyLinkedListNode<T>(
				theElementToAdd, null, this.head
			);
			this.head.setsItsReferenceToThePreviousDoublyLinkedListNodeTo(theDoublyLinkedListNodeForTheElement);
			this.head = theDoublyLinkedListNodeForTheElement;
		}
		
		else {
			
			ADoublyLinkedListNode<T> theCurrentDoublyLinkedListNode = this.head;
			
			while ((theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextDoublyLinkedListNode() != null) &&
				   (theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextDoublyLinkedListNode().providesItsData().compareTo(theElementToAdd) < 0)) {
				theCurrentDoublyLinkedListNode =
					theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextDoublyLinkedListNode();
			}
			
			ADoublyLinkedListNode<T> theDoublyLinkedListNodeForTheElement = new ADoublyLinkedListNode<T>(
				theElementToAdd, null, theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextDoublyLinkedListNode()
			);
			
			if (theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextDoublyLinkedListNode() != null) {
				theDoublyLinkedListNodeForTheElement
					.providesItsReferenceToTheNextDoublyLinkedListNode()
					.setsItsReferenceToThePreviousDoublyLinkedListNodeTo(theDoublyLinkedListNodeForTheElement);
			}
			
			theCurrentDoublyLinkedListNode.setsItsReferenceToTheNextDoublyLinkedListNodeTo(
				theDoublyLinkedListNodeForTheElement
			);
			theDoublyLinkedListNodeForTheElement.setsItsReferenceToThePreviousDoublyLinkedListNodeTo(
				theCurrentDoublyLinkedListNode
			);
			
		}
		
	}
	
	
	public void remove(T theElementToRemove) {
		
		if (this.head == null) {
			throw new ACollectionUnderflowException("Exception: remove for an empty collection was requested.");
		}
		
		while (this.head != null && theElementToRemove.equals(this.head.providesItsData())) {
			this.head = this.head.providesItsReferenceToTheNextDoublyLinkedListNode();
		}
		
		if (this.head == null) {
			return;
		}
		else {
			this.head.setsItsReferenceToThePreviousDoublyLinkedListNodeTo(null);
		}
		
		ADoublyLinkedListNode<T> theCurrentDoublyLinkedListNode = this.head;
		while (theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextDoublyLinkedListNode() != null) {
			if (theElementToRemove.equals(theCurrentDoublyLinkedListNode.providesItsData())) {
				
				theCurrentDoublyLinkedListNode
					.providesItsReferenceToThePreviousDoublyLinkedListNode()
					.setsItsReferenceToTheNextDoublyLinkedListNodeTo(
						theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextDoublyLinkedListNode()
					);
				
				theCurrentDoublyLinkedListNode
				.providesItsReferenceToTheNextDoublyLinkedListNode()
				.setsItsReferenceToThePreviousDoublyLinkedListNodeTo(
					theCurrentDoublyLinkedListNode.providesItsReferenceToThePreviousDoublyLinkedListNode()
				);

			}
			
			theCurrentDoublyLinkedListNode =
				theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextDoublyLinkedListNode();
			
		}
		
		if (theElementToRemove.equals(theCurrentDoublyLinkedListNode.providesItsData())) {
			
			theCurrentDoublyLinkedListNode
			.providesItsReferenceToThePreviousDoublyLinkedListNode()
			.setsItsReferenceToTheNextDoublyLinkedListNodeTo(
				theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextDoublyLinkedListNode()
			);
			
		}
		
	}
	
	
	public String toString() {
		
		String theRepresentationOfThisCollection = "[\n";
		
		ADoublyLinkedListNode<T> theCurrentDoublyLinkedListNode = this.head;
		while (theCurrentDoublyLinkedListNode != null) {
			theRepresentationOfThisCollection += "\t" + theCurrentDoublyLinkedListNode.providesItsData() + "\n";
			theCurrentDoublyLinkedListNode =
				theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextDoublyLinkedListNode();
		}
		
		theRepresentationOfThisCollection += "]";
		
		return theRepresentationOfThisCollection;
		
	}
	
}
