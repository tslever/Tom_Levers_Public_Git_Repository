package Com.TSL.BinaryExpressionTreeUtilities;


import java.util.LinkedList;
import java.util.Stack;
import java.util.Queue;


/**
* @author YINGJIN CUI
* @version 1.0
* since   2020-05
*
* Student name:  Tom Lever
* Completion date: 07/02/21
*
* BXT.txt: the template file of BXT.java
* Student tasks: implement tasks as specified in this file
*
* Note: BXT Represents a binary expression tree.
* BXT can build itself from a postorder expression.  
* It can evaluate and print itself. It also prints an inorder string and a preorder string.  
*/


class BXT {
	
	
   private TreeNode<String> root;
   private String representationOfThisBinaryExpressionTree;

   
   /**
    * BXT() is the zero-parameter constructor for BXT, which sets this tree's reference to a root node to null. 
    */
   
   public BXT() {
	   
      root = null;
      
   }

   
   public void buildTree(String str) {
      // *** Student task ***  
      /* 
	The argument string is in postfix notation. 
        Build the tree as specified in the document 
        *** Enter your code below ***
     */

       String[] theTokensOfThePostFixExpression = str.split(" ");
       
       Stack<TreeNode<String>> theStackOfTrees = new Stack<TreeNode<String>>();
       TreeNode<String> theTreeNodeRepresentingThePresentOperator;
       
       for (int i = 0; i < theTokensOfThePostFixExpression.length; i++) {
    	   
    	   if (!isOperator(theTokensOfThePostFixExpression[i])) {
    		   
    		   theStackOfTrees.push(new TreeNode<String>(theTokensOfThePostFixExpression[i]));
    		   
    	   }
    	   
    	   else {
    		   
    		   theTreeNodeRepresentingThePresentOperator = new TreeNode<String>(theTokensOfThePostFixExpression[i]);
    		   theTreeNodeRepresentingThePresentOperator.setRight(theStackOfTrees.pop());
    		   theTreeNodeRepresentingThePresentOperator.setLeft(theStackOfTrees.pop());
    		   theStackOfTrees.push(theTreeNodeRepresentingThePresentOperator);
    		   
    	   }
    	   
       }
       
       this.root = theStackOfTrees.pop();
 
   }
   
   
   public String display() {
      // *** Student task ***  
      /* 
	Display tree structure. Plese refer to GA2 if you need to refresh your knowledge
        *** Enter your code below ***
     */

	   this.representationOfThisBinaryExpressionTree = "";
	   
	   displayHelper(root, 0);
	   
	   return this.representationOfThisBinaryExpressionTree;
 
   }
   
   
   /**
    * displayHelper is a recursive method. If a provided reference to a node is null, this method does nothing. This
    * method passes to itself a reference to the right child node of the node at the provided reference and the index
    * of the level after a provided level. Each version of this method appends a tab character to a representation of
    * this tree for every level between the first level in the tree and this version's level inclusive. Each version of
    * this method appends the provided node's string. Each version of this method passes itself a reference to the
    * left child node of the node at the provided reference and the index of the level after the provided level. 
    * 
    * @param t
    * @param level
    */

   private void displayHelper(TreeNode<String> t, int level){
      if(t==null) return ;
      displayHelper(t.getRight(), level + 1);
      for(int k = 0; k < level; k++)
    	  this.representationOfThisBinaryExpressionTree += "\t";
      this.representationOfThisBinaryExpressionTree += t.getValue() + "\n";
      displayHelper(t.getLeft(), level + 1); //recurse left
   }
   
   
   /**
    * evaluateTree provides a floating-point number representing the result of performing all the mathematical
    * operations represented by the binary expression tree.
    * 
    * @return
    */
   
   public double evaluateTree() {
      // *** Student task ***  
      /* 
	Do this recursively.  If the node is an operator, recursively evaluate the left child 
        and the right child, and return the result.  Else the node is a number, so it can 
        be converted into a double, and returned. 
        *** Enter your code below ***
     */

       return evaluateTreeWithRootNode(this.root);

   }
   
   
   /**
    * evaluateTreeWithRootNode is a recursive method that provides a floating-point number representing the result of
    * performing all the mathematical operations represented by this binary expression tree.
    * 
    * @param theRootOfTheTree
    * @return
    */
   
   private double evaluateTreeWithRootNode(TreeNode<String> theRootOfTheTree) {
	   
	   String theToken = theRootOfTheTree.getValue();
	   
	   switch (theToken) {
	   
	       case "+":
	    	   return evaluateTreeWithRootNode(theRootOfTheTree.getLeft()) + evaluateTreeWithRootNode(theRootOfTheTree.getRight());
	    	   
	       case "-":
	    	   return evaluateTreeWithRootNode(theRootOfTheTree.getLeft()) - evaluateTreeWithRootNode(theRootOfTheTree.getRight());
	    	   
	       case "*":
	    	   return evaluateTreeWithRootNode(theRootOfTheTree.getLeft()) * evaluateTreeWithRootNode(theRootOfTheTree.getRight());
	    	   
	       case "/":
	    	   return evaluateTreeWithRootNode(theRootOfTheTree.getLeft()) / evaluateTreeWithRootNode(theRootOfTheTree.getRight());
	    	   
	       case "%":
	    	   return evaluateTreeWithRootNode(theRootOfTheTree.getLeft()) % evaluateTreeWithRootNode(theRootOfTheTree.getRight());
	    	   
	       default:
	    	   return Double.parseDouble(theToken);
	   
	   }
	   
   }
   
   
   /**
    * isOperator indicates whether or not a string is an arithmetic operator.
    * 
    * @param thePotentialOperator
    * @return
    */
   
   public boolean isOperator(String thePotentialOperator) {
	   
	   switch (thePotentialOperator) {
	   
		   case "+":
		   case "-":
		   case "*":
		   case "/":
		   case "%":
			   return true;
		   
		   default:
			   return false;
	   
	   }
	   
   }
   
   
   /**
    * enqueueViaPreOrderTraversalTheStringOf is a recursive method. This method enqueues the string in the node at the
    * provided reference. If a provided reference to a node is null, this method does nothing. This method passes
    * itself the reference to the left child node of the node at the provided reference and a reference to the queue in
    * which to enqueue strings. Finally, this method passes itself the reference to the right child node of the node at
    * the provided reference and a reference to the queue.
    * 
    * @param theNode
    * @param theQueue
    */
   
   private void enqueueViaPreOrderTraversalTheStringsOf(
       TreeNode<String> theNodeWithAToken, Queue<String> theQueueOfTokens
   ) {
	   
	   if (theNodeWithAToken == null) {
		   return;
	   }
	   
	   theQueueOfTokens.add(theNodeWithAToken.getValue());
	   enqueueViaPreOrderTraversalTheStringsOf(theNodeWithAToken.getLeft(), theQueueOfTokens);
	   enqueueViaPreOrderTraversalTheStringsOf(theNodeWithAToken.getRight(), theQueueOfTokens);
	   
   }
   
   
   
   /**
    * enqueueViaInOrderTraversalTheStringOf is a recursive method. If a provided reference to a node is null, this
    * method does nothing. This method passes itself the reference to the left child node of the node at the provided
    * reference and a reference to the queue in which to enqueue strings. After this method's descendant returns,
    * this method enqueues the string in the node at the provided reference. Finally, this method passes itself the
    * reference to the right child node of the node at the provided reference and a reference to the queue.
    * 
    * @param theNode
    * @param theQueue
    */
   
   private void enqueueViaInOrderTraversalTheStringsOf(
       TreeNode<String> theNodeWithAToken, Queue<String> theQueueOfTokens
   ) {
	   
	   if (theNodeWithAToken == null) {
		   return;
	   }
	   
	   enqueueViaInOrderTraversalTheStringsOf(theNodeWithAToken.getLeft(), theQueueOfTokens);
	   theQueueOfTokens.add(theNodeWithAToken.getValue());
	   enqueueViaInOrderTraversalTheStringsOf(theNodeWithAToken.getRight(), theQueueOfTokens);
	   
   }
   
   
   /**
    * enqueueViaPostOrderTraversalTheStringOf is a recursive method. If a provided reference to a node is null, this
    * method does nothing. This method passes itself the reference to the left child node of the node at the provided
    * reference and a reference to the queue in which to enqueue strings. This method passes itself the reference to
    * the right child node of the node at the provided reference and a reference to the queue. After this method's
    * descendant returns, this method enqueues the string in the node at the provided reference.
    * 
    * @param theNode
    * @param theQueue
    */
   
   private void enqueueViaPostOrderTraversalTheStringsOf(
       TreeNode<String> theNodeWithAToken, Queue<String> theQueueOfTokens
   ) {
	   
	   if (theNodeWithAToken == null) {
		   return;
	   }
	   
	   enqueueViaPostOrderTraversalTheStringsOf(theNodeWithAToken.getLeft(), theQueueOfTokens);
	   enqueueViaPostOrderTraversalTheStringsOf(theNodeWithAToken.getRight(), theQueueOfTokens);
	   theQueueOfTokens.add(theNodeWithAToken.getValue());
	   
   }
   
   
   /**
    * getTheArithmeticExpressionBasedOn provides an arithmetic expression based on a queue of mathematical tokens.
    * 
    * @param theQueueOfTokens
    * @return
    */
   
   private String getTheArithmeticExpressionBasedOn(Queue<String> theQueueOfTokens) {
	   
	   String theArithmeticExpression = "";
	   
	   int theNumberOfTokens = theQueueOfTokens.size();
	   for (int i = 0; i < theNumberOfTokens - 1; i++) {
		   theArithmeticExpression += theQueueOfTokens.remove() + " ";
	   }
	   theArithmeticExpression += theQueueOfTokens.remove();
	   
	   return theArithmeticExpression;
	   
   }
   
   
   public String infix() { 
      // *** Student task ***  
      /* 
	Infix is characterized by the placement of operators between operands; 
        *** Enter your code below ***
     */
	   
	   Queue<String> theQueueOfTokens = new LinkedList<String>();
	   enqueueViaInOrderTraversalTheStringsOf(this.root, theQueueOfTokens);
	   
	   return getTheArithmeticExpressionBasedOn(theQueueOfTokens);

   }
   
   
   public String postfix(){
   // *** Student task ***  
   /* 
	Postfix requires that its operators come after the corresponding operands
     *** Enter your code below ***
  */

       Queue<String> theQueueOfTokens = new LinkedList<String>();
       enqueueViaPostOrderTraversalTheStringsOf(this.root, theQueueOfTokens);
       
       return getTheArithmeticExpressionBasedOn(theQueueOfTokens);

   }

   
   public String prefix(){
      // *** Student task ***  
      /* 
	Prefix expression notation requires that all operators precede the two operands that they work on; 
        *** Enter your code below ***
     */

       Queue<String> theQueueOfTokens = new LinkedList<String>();
       enqueueViaPreOrderTraversalTheStringsOf(this.root, theQueueOfTokens);
       
       return getTheArithmeticExpressionBasedOn(theQueueOfTokens);

   }

}