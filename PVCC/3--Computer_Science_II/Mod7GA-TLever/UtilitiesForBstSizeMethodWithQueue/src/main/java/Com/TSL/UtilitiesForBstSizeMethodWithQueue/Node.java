package Com.TSL.UtilitiesForBstSizeMethodWithQueue;


/**
* @author Yingjing Cui
* version 1.0
* since   2020-05-24 16:24
*
* Node.java: Node class
*/

public class Node {

	
   private int data;
   private Node left;
   private Node right;

   
   /**
    * Node(int data) is the one-parameter constructor for Node, which sets the value of this Node's data to provided
    * data, sets this Node's reference to a left child node to null, and sets this Node's reference to a right child
    * node to null.
    * 
    * @param data
    */
   
   public Node(int data) {
	   
      this.data =data;
      left=right=null;
      
   }

   
   /**
    * getData provides this Node's data.
    * 
    * @return
    */
   
   public int getData() {
      return data;
   }
   
   
   /**
    * getLeft provides this Node's reference to its left child node.
    * @return
    */
   
   public Node getLeft(){
      return left;
   }
   
   
   /**
    * getRight provides this Node's reference to its right child node. 
    * @return
    */
   
   public Node getRight(){
      return right;
   }
   
   
   /**
    * setLeft sets this Node's reference to its left child node to a provided reference to a node.
    * @param node
    */

   public void setLeft(Node node){
      this.left = node;
   }
   
   
   /**
    * setRight sets this Node's reference to its right child node to a provided reference to a node.
    * @param node
    */
   
   public void setRight(Node node){
      this.right = node;
   }
   
}