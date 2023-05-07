package Com.TSL.UtilitiesForBalancingBinarySearchTree;


/**
* @author Yingjin Cui
* version 1.0
* since   2020-05-24  
*
* GA2Driver.java: the driver program for MyIntBSTTree class
*/

public class GA2Driver {

	
   /**
    * main is the entry point of this program, which adds nodes to a binary search tree, displays information about and
    * a representation of the tree, balances the tree, and provides information about and a representation of the tree.
    * 
    * @param args
    */
	
   public static void main(String[] args){
      
      MyIntBSTTree tree=new MyIntBSTTree();
      
      tree.add(1);
      tree.add(3);
      tree.add(4);
      tree.add(12);
      tree.add(20);
      tree.add(26);
      tree.add(68);
      
      System.out.println("Before balancing:");
      System.out.println("Tree height: "+ tree.height());
      tree.display();
      
      System.out.println("\n\nAfter balancing:");
      tree = tree.balance();
      System.out.println("Tree height: "+ tree.height());
      tree.display();
   }

}