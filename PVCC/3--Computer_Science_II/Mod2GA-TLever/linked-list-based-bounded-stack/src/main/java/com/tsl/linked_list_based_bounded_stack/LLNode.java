package com.tsl.linked_list_based_bounded_stack;


/**
 * LLNode provides a structure for objects that represent linked-list nodes.
 * @author Tom Lever
 * @version 1.0
 * @since 05/28/21
 * @param <T>
 */

public class LLNode<T> {


	private T information;
	/**
	 * information is a component of this linked-list node.
	 */
	
	
	private LLNode<T> referenceToTheNextLinkedListNode;
	/**
	 * referenceToTheNextLinkedListNode is a component of this linked-list node.
	 */	
	
	
	public LLNode(T theInformationToUse) {
	/**
	 * LLNode(T theInformationToUse) is a one-parameter constructor for LLNode that sets this linked-list node's
	 * information to theInformationToUse and referenceToTheNextLinkedListNode to null.
	 * @param theInformationToUse
	 */
		
		this.information = theInformationToUse;
		this.referenceToTheNextLinkedListNode = null;
		
	}
	

	public void setsItsInformationTo(T theInformationToUse) {
	/**
	 * setsItsInformationTo sets this linked-list node's information to theInformationToUse.
	 * @param theInformationToUse
	 */
		
		this.information = theInformationToUse;
		
	}
	
	

	public T getTheInformation() {
	/**
	 * getTheInformation provides this linked-list node's information.
	 * @return
	 */
		
		return this.information;
		
	}
	

	public void setsItsReferenceTo(LLNode<T> theReference) {
	/**
	 * setsItsReferenceTo sets this linked-list node's reference to TheNext linked-list node to theReference.
	 * @param theReference
	 */
		
		this.referenceToTheNextLinkedListNode = theReference;
		
	}
	

	public LLNode<T> getTheReferenceToTheNextLinkedListNode() {
	/**
	 * references provides the linked-list node that this linked-list node references.
	 * @return
	 */
		
		return this.referenceToTheNextLinkedListNode;
		
	}
	
}
