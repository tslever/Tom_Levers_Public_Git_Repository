package Com.TSL.LineEditorUtilities;


/**
 * Node represents the structure for a singly linked list node.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/27/21
 *
 * @param <T>
 */

public class Node<T> {

	
	private T data;
	private Node<T> referenceToTheNextNode;
	
	
	/**
	 * Node(T theDataToUse, Node<T> theReferenceToTheNextNodeToUse) is the two-parameter constructor for Node, which
	 * sets this node's data to theDataToUse and sets this node's reference to the next node to
	 * theReferenceToTheNextNodeToUse.
	 * 
	 * @param theDataToUse
	 * @param theReferenceToTheNextNodeToUse
	 */
	
	public Node(T theDataToUse, Node<T> theReferenceToTheNextNodeToUse) {
		
		this.data = theDataToUse;
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
	 * providesItsReferenceToTheNextNode provides this node's reference to the next node.
	 * 
	 * @return
	 */
	
	public Node<T> providesItsReferenceToTheNextNode() {
		return this.referenceToTheNextNode;
	}
	
	
	/**
	 * setsItsDataTo sets this node's data to provided data.
	 * 
	 * @param theDataToUse
	 */
	
	public void setsItsDataTo(T theDataToUse) {
		this.data = theDataToUse;
	}
	
	
	/**
	 * setsItsReferenceToTheNextNodeTo sets this node's reference to the next node to theReferenceToTheNextNodeToUse.
	 * 
	 * @param theReferenceToTheNextNodeToUse
	 */
	
	public void setsItsReferenceToTheNextNodeTo(Node<T> theReferenceToTheNextNodeToUse) {
		this.referenceToTheNextNode = theReferenceToTheNextNodeToUse;
	}
	
}