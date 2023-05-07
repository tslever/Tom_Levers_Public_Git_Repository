package Com.TSL.StockTransactionUtilities;


/**
 * ALinkedListNode represents the structure for a linked-list node.
 * 
 * @author Tom
 * @version 1.0
 * @since 06/11/21
 * @param <T>
 */

public class ALinkedListNode<T> {
	
	private ALinkedListNode<T> reference;
	private T element;
	
	
	/**
	 * ALinkedListNode(T theElementToUse) is the one-parameter constructor for ALinkedListNode, which sets this
	 * linked-list node's element to a given element.
	 * 
	 * @param theElementToUse
	 */
	
	public ALinkedListNode(T theElementToUse) {
		
		this.element = theElementToUse;
		this.reference = null;
		
	}
	
	
	/**
	 * setsItsElementTo sets the element of this linked-list node to a given element.
	 * 
	 * @param theElementToUse
	 */
	
	public void setsItsElementTo(T theElementToUse) {
		
		this.element = theElementToUse;
		
	}
	
	
	/**
	 * providesItsElement provides the element of this linked-list node.
	 * 
	 * @return
	 */
	
	public T providesItsElement() {
		
		return this.element;
		
	}
	
	
	/**
	 * setsItsReferenceTo sets the reference of this linked-list node to a given linked-list node.
	 * 
	 * @param theReferenceToUse
	 */
	
	public void setsItsReferenceTo(ALinkedListNode<T> theReferenceToUse) {
		
		this.reference = theReferenceToUse;
		
	}
	
	
	/**
	 * providesItsReference provides the reference of this linked-list node.
	 * 
	 * @return
	 */
	
	public ALinkedListNode<T> providesItsReference() {
		
		return this.reference;
		
	}
	
	
}
