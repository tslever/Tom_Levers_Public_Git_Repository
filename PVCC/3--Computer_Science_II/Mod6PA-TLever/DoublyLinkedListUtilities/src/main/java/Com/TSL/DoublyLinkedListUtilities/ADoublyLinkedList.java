package Com.TSL.DoublyLinkedListUtilities;


/**
 * ADoublyLinkedList represents the structure for a doubly linked list.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/24/21
 *
 * @param <T>
 */

public class ADoublyLinkedList<T extends Comparable<T>> {

	
	private ADoublyLinkedListNode<T> head;
	private ADoublyLinkedListNode<T> currentDoublyLinkedListNode;
	private ADoublyLinkedListNode<T> tail;
	private int numberOfElements;
	
	
	/**
	 * ADoublyLinkedList() is the zero-parameter constructor for ADoublyLinkedList, which sets this list's reference
	 * to its first node to null, sets this list's reference to its current node to null, sets this list's reference
	 * to its last node to null, and sets this list's number of elements to 0.
	 */
	
	public ADoublyLinkedList() {
		
		this.head = null;
		this.currentDoublyLinkedListNode = null;
		this.tail = null;
		this.numberOfElements = 0;
		
	}
	
	
	/**
	 * addsToItsHead adds a provided element as the first element of this list.
	 * @param theElement
	 */
	
	public void addsToItsHead(T theElement) {
		
		if (theElement == null) {
			throw new RuntimeException(
				"addToItsHead of a doubly linked list was requested with a reference to an element of null."
			);
		}
		
		if (this.head == null) {
			this.head = new ADoublyLinkedListNode<T>(theElement, null, null);
			this.tail = this.head;
		}
		
		else {
			ADoublyLinkedListNode<T> theDoublyLinkedListNodeForTheElement =
				new ADoublyLinkedListNode<T>(theElement, null, this.head);
			this.head.setsItsReferenceToThePreviousNodeTo(theDoublyLinkedListNodeForTheElement);
			this.head = theDoublyLinkedListNodeForTheElement;
		}
		
		this.numberOfElements++;
		
	}
	
	
	/**
	 * addsToItsTail adds a provided element as the last element of this list.
	 * @param theElement
	 */
	
	public void addsToItsTail(T theElement) {
		
		if (theElement == null) {
			throw new RuntimeException(
				"addToItsHead of a doubly linked list was requested with a reference to an element of null."
			);
		}
		
		if (this.tail == null) {
			this.head = new ADoublyLinkedListNode<T>(theElement, null, null);
			this.tail = this.head;
		}
		
		else {		
			ADoublyLinkedListNode<T> theDoublyLinkedListNodeForTheElement =
				new ADoublyLinkedListNode<T>(theElement, this.tail, null);
			this.tail.setsItsReferenceToTheNextNodeTo(theDoublyLinkedListNodeForTheElement);
			this.tail = this.tail.providesItsReferenceToTheNextNode();
		}
		
		this.numberOfElements++;
		
	}
	
	
	/**
	 * providesItsNumberOfElements provides this list's number of elements.
	 * @return
	 */
	
	public int providesItsNumberOfElements() {
		
		return this.numberOfElements;
		
	}
	
	
	/**
	 * insertsAtItsMidpoint inserts a provided element into the middle of this list. If the number of elements in this
	 * list is odd, the provided element is inserted before the middle element.
	 * 
	 * @param theElement
	 */
	
	public void insertsAtItsMidpoint(T theElement) {
		
		if (theElement == null) {
			throw new RuntimeException(
				"addToItsHead of a doubly linked list was requested with a reference to an element of null."
			);
		}
		
		if (this.head == null) {
			this.head = new ADoublyLinkedListNode<T>(theElement, null, null);
			this.tail = this.head;
		}
		
		else {
		
			ADoublyLinkedListNode<T> theCurrentDoublyLinkedListNode = this.head;
			int theIndexOfTheCurrentNode = 0;
			
			while (theIndexOfTheCurrentNode < (int)((double)this.numberOfElements / 2.0)) {
				theCurrentDoublyLinkedListNode = theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode();
				theIndexOfTheCurrentNode++;
			}
			
			if (theCurrentDoublyLinkedListNode == this.head) {
				addsToItsHead(theElement);
			}
			
			else {
			
				ADoublyLinkedListNode<T> theDoublyLinkedListNodeForTheElement = new ADoublyLinkedListNode<T>(
					theElement,
					theCurrentDoublyLinkedListNode.providesItsReferenceToThePreviousNode(),
					theCurrentDoublyLinkedListNode
				);
				
				theCurrentDoublyLinkedListNode.providesItsReferenceToThePreviousNode().setsItsReferenceToTheNextNodeTo(
					theDoublyLinkedListNodeForTheElement
				);
					
				theCurrentDoublyLinkedListNode.setsItsReferenceToThePreviousNodeTo(
					theDoublyLinkedListNodeForTheElement
				);
			
			}
		
		}
		
		this.numberOfElements++;
		
	}
	
	
	/**
	 * removesTheNodeAtItsHead removes the first node of this doubly linked list.
	 */
	
	public void removesTheNodeAtItsHead() {
		
		if (this.head == null) {
			throw new RuntimeException("There is no node in the doubly linked list to remove.");
		}
		
		this.head = this.head.providesItsReferenceToTheNextNode();
		
		this.numberOfElements--;
		
	}
	
	
	/**
	 * setsItsCurrentNodeToItsTail sets the reference to the current node of this list to the reference to the last
	 * node of this list. 
	 */
	
	public void setsItsCurrentNodeToItsTail() {
		
		this.currentDoublyLinkedListNode = this.tail;
		
	}
	
	
	/**
	 * providesTheElementOfItsCurrentNode provides the element of the current node of this list.
	 * 
	 * @return
	 */
	
	public T providesTheElementOfItsCurrentNode() {
		
		if (this.currentDoublyLinkedListNode == null) {
			throw new RuntimeException(
				"providesTheElementOfItsCurrentNode of a doubly linked list was requested when the list's reference " +
				"to a current node was null."
			);
		}
		
		return this.currentDoublyLinkedListNode.providesItsData();
		
	}
	
	
	/**
	 * movesItsCurrentNodeAwayFromItsTailAndTowardsItsHead resets the reference to the current node of this list
	 * as the reference to the node before the current node of this list.
	 */
	
	public void movesItsCurrentNodeAwayFromItsTailAndTowardsItsHead() {
		
		if (this.currentDoublyLinkedListNode == null) {
			throw new RuntimeException("The reference to the current node of this doubly linked list is null.");
		}
		
		if (this.currentDoublyLinkedListNode.providesItsReferenceToThePreviousNode() == null) {
			throw new RuntimeException(
				"The reference to the current node of this doubly linked list would become null."
			);
		}
		
		this.currentDoublyLinkedListNode = this.currentDoublyLinkedListNode.providesItsReferenceToThePreviousNode();
		
	}
	
	
	/**
	 * removesTheNodeAtItsMidpoint removes the node at the midpoint of this list. If the number of nodes in this list
	 * is even, the node pointed to by the middle link is removed.
	 */
	
	public void removesTheNodeAtItsMidpoint() {
		
		if (this.head == null) {
			throw new RuntimeException("There is no node in the doubly linked list to remove.");
		}
		
		ADoublyLinkedListNode<T> theCurrentDoublyLinkedListNode = this.head;
		int theIndexOfTheCurrentNode = 0;
		
		while (theIndexOfTheCurrentNode < (int)((double)this.numberOfElements / 2.0)) {
			theCurrentDoublyLinkedListNode = theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode();
			theIndexOfTheCurrentNode++;
		}
		
		if (theCurrentDoublyLinkedListNode == this.head) {
			this.head = null;
		}
		
		else {
			theCurrentDoublyLinkedListNode.providesItsReferenceToThePreviousNode().setsItsReferenceToTheNextNodeTo(
				theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode()
			);
		}
		
		this.numberOfElements--;
		
	}
	
	
	/**
	 * finds provides the first element in this list that equals a provided element.
	 * 
	 * @param theElement
	 * @return
	 */
	
	public T finds(T theElement) {
		
		ADoublyLinkedListNode<T> theCurrentDoublyLinkedListNode = this.head;
		
		while (theCurrentDoublyLinkedListNode != null) {
			if (theCurrentDoublyLinkedListNode.providesItsData().equals(theElement)) {
				return theCurrentDoublyLinkedListNode.providesItsData();
			}
			theCurrentDoublyLinkedListNode = theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode();
		}
		
		return null;
		
	}
	
	
	/**
	 * toString provides a string representation of this list.
	 */
	
	public String toString() {
		
		String theRepresentationOfThisDoublyLinkedList = "[";
		
		ADoublyLinkedListNode<T> theCurrentDoublyLinkedListNode = this.head;
		while (theCurrentDoublyLinkedListNode != null) {
			theRepresentationOfThisDoublyLinkedList += "\n\t" + theCurrentDoublyLinkedListNode.providesItsData();
			theCurrentDoublyLinkedListNode = theCurrentDoublyLinkedListNode.providesItsReferenceToTheNextNode();
		}
		
		theRepresentationOfThisDoublyLinkedList += "\n]";
		
		return theRepresentationOfThisDoublyLinkedList;
		
	}
	
	
	/**
	 * ADoublyLinkedListNode represents the structure for a doubly linked list node.
	 * 
	 * @author Tom Lever
	 * @version 1.0
	 * @since 06/24/21
	 *
	 * @param <T>
	 */
	
	private class ADoublyLinkedListNode<T> {
		
		private T data;
		private ADoublyLinkedListNode<T> referenceToThePreviousNode;
		private ADoublyLinkedListNode<T> referenceToTheNextNode;
		
		
		/**
		 * ADoublyLinkedListNode(...) is the three parameter constructor for ADoublyLinkedListNode, which sets this
		 * node's data to theDataToUse, this node's reference to the previous node to
		 * theReferenceToThePreviousNodeToUse, and this node's reference to the next node to
		 * theReferenceToTheNextNodeToUse.
		 * 
		 * @param theDataToUse
		 * @param theReferenceToThePreviousNodeToUse
		 * @param theReferenceToTheNextNodeToUse
		 */
		
		public ADoublyLinkedListNode(
			T theDataToUse,
			ADoublyLinkedListNode<T> theReferenceToThePreviousNodeToUse,
			ADoublyLinkedListNode<T> theReferenceToTheNextNodeToUse
		) {
			
			this.data = theDataToUse;
			this.referenceToThePreviousNode = theReferenceToThePreviousNodeToUse;
			this.referenceToTheNextNode = theReferenceToTheNextNodeToUse;
			
		}
		
		
		/**
		 * setsItsReferenceToThePreviousNodeTo sets this node's reference to the previous node to
		 * theReferenceToThePreviousNodeToUse
		 * 
		 * @param theReferenceToThePreviousNodeToUse
		 */
		
		public void setsItsReferenceToThePreviousNodeTo(ADoublyLinkedListNode<T> theReferenceToThePreviousNodeToUse) {
			this.referenceToThePreviousNode = theReferenceToThePreviousNodeToUse;
		}
		
		
		/**
		 * setsItsReferenceToTheNextNodeTo sets this node's reference to the next node to
		 * theReferenceToTheNextNodeToUse.
		 * 
		 * @param theReferenceToTheNextNodeToUse
		 */
		
		public void setsItsReferenceToTheNextNodeTo(ADoublyLinkedListNode<T> theReferenceToTheNextNodeToUse) {
			this.referenceToTheNextNode = theReferenceToTheNextNodeToUse;
		}

		
		/**
		 * providesItsData provides this node's data.
		 * 
		 * @return
		 */
		
		public T providesItsData() {
			return this.data;
		}

		
		/**
		 * providesItsReferenceToThePreviousNode provides this node's reference to the previous node.
		 * 
		 * @return
		 */
		
		public ADoublyLinkedListNode<T> providesItsReferenceToThePreviousNode() {
			return this.referenceToThePreviousNode;
		}
		
		
		/**
		 * providesItsReferenceToTheNextNode provides this node's reference to the next node.
		 * 
		 * @return
		 */
		
		public ADoublyLinkedListNode<T> providesItsReferenceToTheNextNode() {
			return this.referenceToTheNextNode;
		}
		
	}
	
}
