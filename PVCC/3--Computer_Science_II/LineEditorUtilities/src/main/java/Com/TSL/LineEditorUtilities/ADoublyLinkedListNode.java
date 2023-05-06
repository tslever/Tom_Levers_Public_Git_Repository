package Com.TSL.LineEditorUtilities;


/**
 * A DoublyLinkedListNode represents the structure for a doubly linked list node.
 * @author Tom Lever
 * @version 1.0
 * @since 06/24/21
 *
 * @param <T>
 */

public class ADoublyLinkedListNode<T> {

	
	private T data;
	private ADoublyLinkedListNode<T> referenceToThePreviousNode;
	private ADoublyLinkedListNode<T> referenceToTheNextNode;
	
	
	/**
	 * ADoublyLinkedListNode(...) is the three parameter constructor for ADoublyLinkedListNode, which sets this node's
	 * data to theDataToUse, this node's reference to the previous node to theReferenceToThePreviousNodeToUse, and
	 * this node's reference to the next node to theReferenceToTheNextNodeToUse.
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