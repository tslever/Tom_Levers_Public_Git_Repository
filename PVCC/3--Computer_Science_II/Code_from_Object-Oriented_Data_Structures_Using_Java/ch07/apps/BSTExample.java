//--------------------------------------------------------------------------
// BSTExample.java          by Dale/Joyce/Weems                    Chapter 7
//
// Creates a BST to match Figure 7.8 and demonstrates use of iteration.
//--------------------------------------------------------------------------
package ch07.apps;

import ch07.trees.*;
import java.util.Iterator; 

public class BSTExample 
{
  public static void main(String[] args)
  {  
    BinarySearchTree<Character> example = new BinarySearchTree<Character>();
    Iterator<Character> iter;
    
    example.add('P'); example.add('F'); example.add('S'); example.add('B'); 
    example.add('H'); example.add('R'); example.add('Y'); example.add('G'); 
    example.add('T'); example.add('Z'); example.add('W');  

    // Inorder
    System.out.print("Inorder:   ");
    iter = example.getIterator(BSTInterface.Traversal.Inorder);
    while (iter.hasNext())
      System.out.print(iter.next());

    // Preorder
    System.out.print("\nPreorder:  ");
    iter = example.getIterator(BSTInterface.Traversal.Preorder);
    while (iter.hasNext())
      System.out.print(iter.next());

    // Postorder
    System.out.print("\nPostorder: ");
    iter = example.getIterator(BSTInterface.Traversal.Postorder);
    while (iter.hasNext())
      System.out.print(iter.next());

    // Inorder again
    System.out.print("\nInorder:   ");
    for (Character ch: example)
       System.out.print(ch);

    // Inorder again
    System.out.print("\nInorder:   ");
    iter = example.getIterator(BSTInterface.Traversal.Inorder);
    example.add('A'); example.add('A'); example.add('A'); 
    while (iter.hasNext())
      System.out.print(iter.next());

    // Inorder again
    System.out.print("\nInorder:   ");
    iter = example.getIterator(BSTInterface.Traversal.Inorder);
    while (iter.hasNext())
      System.out.print(iter.next());
  }
}