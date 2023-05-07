package Com.TSL.UtilitiesForBstSizeMethodWithQueue;


/**
* @author Yingjin Cui
* version 1.0
* since   2020-05-24 16:55
*
* Student name:  Tom Lever
* Completion date: 07/02/21
*
* MyIntBSTTree represents the structure of a binary search tree whose nodes contain integers.
*/

import java.util.*;

public class MyIntBSTTree{


   private Node root;

   
   /**
    * MyIntBSTTree() is the zero-parameter constructor for MyIntBSTTree, which sets this
    * tree's reference to a root node to null.
    */
   
   public MyIntBSTTree() {
	   
      root=null;
      
   }
   

   public int size() {
 
      // *** Student task #1 ***  
      /* 
      Requirements: 
      - Implement this method with a queue.
      - This method returns the number of elements in the tree
      
       *** Enter your code below *** 
     */
      
      if (root == null) {
    	  return 0;
      }
      
      int count = 0;
      Queue<Node> queue = new LinkedList<Node>();
	  Node thePresentNode;
	  queue.add(root);
	  
	  while (!queue.isEmpty()) {
		  thePresentNode = queue.remove();
		  count++;
		  if (thePresentNode.getLeft() != null) {
			  queue.add(thePresentNode.getLeft());
		  }
		  if (thePresentNode.getRight() != null) {
			  queue.add(thePresentNode.getRight());
		  }
	  }
	  
	  return count;
	  
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
           node.setRight(addHelper(node.getRight(), data)); //System.out.println(data);
       }
       return node;
   }
   
   
   /**
    * enqueueInOrderTheIntegersOf is a recursive method. If a provided reference to a node is null, this method does
    * nothing. This method passes itself the reference to the left child node of the provided reference to a node and
    * a reference to the queue in which to enqueue integers in ascending order. After this method's descendant returns,
    * this method enqueues the integer in the node referenced by the provided reference to a node. Finally, this
    * method passes itself the reference to the right child node of the provided reference to a node and a reference
    * to the queue.
    * 
    * @param theNode
    * @param theQueue
    */
   
   private void enqueueInOrderTheIntegersOf(Node theNode, Queue<Integer> theQueue) {
	   
	   if (theNode == null) {
		   return;
	   }
	   
	   enqueueInOrderTheIntegersOf(theNode.getLeft(), theQueue);
	   theQueue.add(theNode.getData());
	   enqueueInOrderTheIntegersOf(theNode.getRight(), theQueue);
	   
   }
   
   
   public void printInOrder(){

      // *** Student task #2 ***  

      /* 
      Requirements: 
      - Print all elements in the tree in ascending order. 
      - For example, if the tree contains nodes with values 5, 2, 8, 
        then calling printInOrder() should print as follows:
        [5, 2, 8]
      - You may implement this method either recursively or iteratively.

       *** Enter your code below *** 
     */
	   
      if (root == null) {
    	  System.out.println("[]");
    	  return;
      }
      
      Queue<Integer> theQueue = new LinkedList<Integer>();
      enqueueInOrderTheIntegersOf(root, theQueue);
      
      String theRepresentationOfAnArrayOfTheIntegersInThisBst = "[";
      
      int thePresentInteger;
      while (!theQueue.isEmpty()) {
    	  thePresentInteger = theQueue.remove();
    	  theRepresentationOfAnArrayOfTheIntegersInThisBst += Integer.toString(thePresentInteger);
    	  if (!theQueue.isEmpty()) {
    		  theRepresentationOfAnArrayOfTheIntegersInThisBst += ", ";
    	  }
      }
      
      theRepresentationOfAnArrayOfTheIntegersInThisBst += "]";
      
      System.out.println(theRepresentationOfAnArrayOfTheIntegersInThisBst);
      
   }
   
}