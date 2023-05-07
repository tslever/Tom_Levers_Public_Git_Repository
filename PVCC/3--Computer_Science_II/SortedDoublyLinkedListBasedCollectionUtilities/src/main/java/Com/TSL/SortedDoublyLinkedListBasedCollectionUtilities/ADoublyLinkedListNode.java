package Com.TSL.SortedDoublyLinkedListBasedCollectionUtilities;


public class ADoublyLinkedListNode<T> {

	
	private T data;
	private ADoublyLinkedListNode<T> referenceToThePreviousDoublyLinkedListNode;
	private ADoublyLinkedListNode<T> referenceToTheNextDoublyLinkedListNode;
	
	
	public ADoublyLinkedListNode(
		T theDataToUse,
		ADoublyLinkedListNode<T> theReferenceToThePreviousDoublyLinkedListNode,
		ADoublyLinkedListNode<T> theReferenceToTheNextDoublyLinkedListNode
	) {
		
		this.data = theDataToUse;
		this.referenceToThePreviousDoublyLinkedListNode = theReferenceToThePreviousDoublyLinkedListNode;
		this.referenceToTheNextDoublyLinkedListNode = theReferenceToTheNextDoublyLinkedListNode;
		
	}
	
	
	public T providesItsData() {
		return this.data;
	}
	
	
	public ADoublyLinkedListNode<T> providesItsReferenceToThePreviousDoublyLinkedListNode() {
		return this.referenceToThePreviousDoublyLinkedListNode;
	}
	

	public ADoublyLinkedListNode<T> providesItsReferenceToTheNextDoublyLinkedListNode() {
		return this.referenceToTheNextDoublyLinkedListNode;
	}
	
	
	public void setsItsReferenceToTheNextDoublyLinkedListNodeTo(
		ADoublyLinkedListNode<T> theReferenceToADoublyLinkedListNode
	) {
		this.referenceToTheNextDoublyLinkedListNode = theReferenceToADoublyLinkedListNode;
	}
	
	
	public void setsItsReferenceToThePreviousDoublyLinkedListNodeTo(
		ADoublyLinkedListNode<T> theReferenceToADoublyLinkedListNode
	) {
		this.referenceToThePreviousDoublyLinkedListNode = theReferenceToADoublyLinkedListNode;
	}
	
}
