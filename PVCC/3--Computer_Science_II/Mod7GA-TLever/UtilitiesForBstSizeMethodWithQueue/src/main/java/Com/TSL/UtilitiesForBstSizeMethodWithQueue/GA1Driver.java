package Com.TSL.UtilitiesForBstSizeMethodWithQueue;


/**
* @author YINGJIN CUI
* @version 1.0
* since   2020-05
*
* GA1Driver.java: the driver program for MyIntBSTTree.java
*/

public class GA1Driver{

	
   /**
    * main represents the entry point of this program, which adds integers to a binary search tree with nodes containing
    * integers and prints information about the tree.
    * 
    * @param args
    */
	
   public static void main(String[] args){
      
      MyIntBSTTree tree = new MyIntBSTTree();
      
      tree.add(8);
      tree.add(30);
      tree.add(28);
      tree.add(12);
      tree.add(1);
      tree.add(6);
      
      System.out.println("Number of elements: "+ tree.size());
      tree.printInOrder();
   }

}