package Com.TSL.UtilitiesForBalancingBinarySearchTree;


/**
* @author Yingjin Cui
* version 1.0
* since   2020-05-24 16:55
*
* Student name:  Tom Lever
* Completion date: 07/02/21
*
* MyIntBSTTree represents the structure of a binary search tree with nodes of integers.
*/

import java.util.*;

public class MyIntBSTTree{

	
   private Node root;

   
   /**
    * MyIntBSTTree() is the zero-parameter constructor for MyIntBSTTree, which sets this tree's reference to a root
    * node to null.
    */
   
   public MyIntBSTTree(){
	   
      root=null;
      
   }
   
   
   /**
    * getTheHeightOfTheTreeWithRootNode provides the height of a binary search tree with a provided root.
    * 
    * @param theRootOfTheTree
    * @return
    */
   
   private int getTheHeightOfTheTreeWithRootNode(Node theRootOfTheTree) {
	   
	   if (theRootOfTheTree == null) {
		   return -1;
	   }
	   
	   return
	       1 + Math.max(
               getTheHeightOfTheTreeWithRootNode(theRootOfTheTree.getLeft()),
               getTheHeightOfTheTreeWithRootNode(theRootOfTheTree.getRight())
           );
	   
   }
   
   
   public int height() {
 
      // *** Student task #1 ***  

      /* Requirements: 
         The height of a binary tree is the largest number of edges in a path from the root node to a leaf node. 
         Essentially, it is the height of the root node. Note that if a tree has only one node, then that node 
         is at the same time the root node and the only leaf node, so the height of the tree is 0, similary, 
         the height of a tree with only two nodes is 1. 
         - Implement this method to return height of the tree
       
       *** Enter your code below *** 
     */

	   return getTheHeightOfTheTreeWithRootNode(this.root);

   }
   
   
   /**
    * enqueueInOrder is a recursive method. If a provided reference to a node is null, this method does nothing. This
    * method passes itself the reference to the left child node of the provided reference to a node and a reference to
    * the queue in which to enqueue nodes with integers in ascending order. After this method's descendant returns,
    * this method enqueues the node referenced by the provided reference to a node. Finally, this method passes itself
    * the reference to the right child node of the provided reference to a node and a reference to the queue.
    * 
    * @param theNode
    * @param theQueue
    */
   
   private void enqueueInOrder(Node theNode, Queue<Node> theQueue) {
	   
	   if (theNode == null) {
		   return;
	   }
	   
	   enqueueInOrder(theNode.getLeft(), theQueue);
	   theQueue.add(theNode);
	   enqueueInOrder(theNode.getRight(), theQueue);
	   
   }

    
   public Node[] inOrderArray() {
  
      // *** Student task #2 ***  

      /* Requirements: 
         This method get all elements in the tree and return them as sorted Node array
       
       *** Enter your code below *** 
     */

       if (root == null) {
    	   return new Node[0];
       }
       
       Queue<Node> theQueue = new LinkedList<Node>();
       enqueueInOrder(this.root, theQueue);
       
       return theQueue.toArray(new Node[theQueue.size()]);

   }
   
   
   /**
    * grow grows a binary tree with elements of a sorted array of nodes with integers between the first index of the
    * array and the last index of the array inclusive.
    * 
    * @param theTreeToGrow
    * @param theSortedArrayOfNodes
    * @param theIndexOfTheFirstElement
    * @param theIndexOfTheLastElement
    */
   
   private void grow(
       MyIntBSTTree theTreeToGrow,
       Node[] theSortedArrayOfNodes,
       int theIndexOfTheFirstElement,
       int theIndexOfTheLastElement
   ) {
	   if (theIndexOfTheFirstElement == theIndexOfTheLastElement) {
		   theTreeToGrow.add(theSortedArrayOfNodes[theIndexOfTheFirstElement].getData());
	   }
	   else if ((theIndexOfTheFirstElement + 1) == theIndexOfTheLastElement) {
		   theTreeToGrow.add(theSortedArrayOfNodes[theIndexOfTheFirstElement].getData());
		   theTreeToGrow.add(theSortedArrayOfNodes[theIndexOfTheLastElement].getData());
	   }
	   else {
		   int theIndexOfTheMiddleElement = (theIndexOfTheFirstElement + theIndexOfTheLastElement) / 2;
		   theTreeToGrow.add(theSortedArrayOfNodes[theIndexOfTheMiddleElement].getData());
		   grow(theTreeToGrow, theSortedArrayOfNodes, theIndexOfTheFirstElement, theIndexOfTheMiddleElement - 1);
		   grow(theTreeToGrow, theSortedArrayOfNodes, theIndexOfTheMiddleElement + 1, theIndexOfTheLastElement);
	   }
   }
   

   public MyIntBSTTree balance() { 

      // *** Student task #3 ***  
      /* Requirements: 
         This method rebuilds tree to minimize the level (height) of the tree.
         To do so, you are going to rebuild a new tree from the ordered node elelemts array.
         To minimize the height of the tree, for any node, you try to keep balanced numbers 
         of it's left and right subtrees. Please following the steps to achieve this goal:
         1. select and add the middle element of the array,the middle element divides the
            arry into two parts: part1-(before middle one) and part2-(after the middle one)
         2. For part1 and part2, go to step 1. Repet the process until all elements are added.
            For example, for an array {1,3,6,8,9,12,20}, add 8 to tree, the middle value 8 divides
            the array into two parts: Part 1: {1,3,6} and Part 2: {9,12,20}, for part 1, add 3, 
            for part 2, add 12, repeat the process until all elements are added.
         3. Return the newly builded tree.
       
       *** Enter your code below *** 
     */

	   Node[] theSortedArrayOfNodes = inOrderArray();
       
       MyIntBSTTree theBalancedBinarySearchTree = new MyIntBSTTree();
       
       grow(theBalancedBinarySearchTree, theSortedArrayOfNodes, 0, theSortedArrayOfNodes.length - 1);
       
       return theBalancedBinarySearchTree;
 
   }

   
   /**
    * add adds provided data to this tree.
    * @param data
    */
     
   public void add(int data) {
       root = addHelper(root, data);
   }
   
   
   /**
    * addHelper is a recursive method. If a provided node variable contains a null reference, addHelper sets the
    * variable to reference a node with the provided data. Else, if the provided integer data is less than the data of
    * the provided node, then addHelper passes to itself the reference to the provided node's left child and the
    * provided data. Else, addHelper passes to itself the reference to the provided node's right child and the provided
    * data. addHelper returns the node variable with an updated reference. 
    * 
    * @param node
    * @param data
    * @return
    */

   private Node addHelper(Node node, int data) {//add node helper
       if (node == null){
          node = new Node(data);
       }else if (data <= node.getData()){
           node.setLeft(addHelper(node.getLeft(), data));
       }else{
           node.setRight(addHelper(node.getRight(), data));//System.out.println(data);
       }
       return node;
   }
   
   
   public void display(){
      //print tree structure
      displayHelper(root, 0);
   }
   
   
   /**
    * displayHelper is a recursive method. If a provided reference to a node is null, this method does nothing. This
    * method passes to itself a reference to the right child node of the node at the provided reference and the index
    * of the level after a provided level. Each version of this method outputs a tab character to the standard output
    * stream for every level between the first level in the tree and this version's level inclusive. Each version of
    * this method outputs the provided node's integer. Each version of this method passes itself a reference to the
    * left child node of the node at the provided reference and the index of the level after the provided level. 
    * 
    * @param t
    * @param level
    */

   private void displayHelper(Node t, int level){
      if(t==null) return ;
      displayHelper(t.getRight(), level + 1);
      for(int k = 0; k < level; k++)
         System.out.print("\t");
      System.out.println(t.getData());
      displayHelper(t.getLeft(), level + 1); //recurse left
   } 
   
   
   /**
    * size provides the number of nodes in this tree.
    * 
    * @return
    */

   public int size(){
      return sizeHelper(root);
   }
   
   
   /**
    * sizeHelper is a recursive method. If a provided reference to a node is null, this method provides a number of
    * nodes of 0. Otherwise, this method provides the sum of 1, the output of itself when passed a reference to the
    * left child node of the node referenced by the provided reference, and the output of itself when passed a
    * reference to the right child node referenced by the provided reference.
    * 
    * @param node
    * @return
    */

   private int sizeHelper(Node node){
      if(node==null) return 0;
      else return 1+sizeHelper(node.getLeft())+sizeHelper(node.getRight());
   }
   
}